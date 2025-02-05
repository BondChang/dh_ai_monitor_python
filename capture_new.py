import cv2
import time
import os
import configparser
import ipaddress
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import logging
from logging.handlers import RotatingFileHandler

from ai_detect import process_images

# ---------------------------- 日志配置 ----------------------------
log_file = 'capture.log'
max_log_size = 5 * 1024 * 1024  # 5MB
backup_count = 500

handler = RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=backup_count)
logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---------------------------- 配置加载 ----------------------------
config = configparser.ConfigParser()
config.read('config.ini')

# RTSP配置
ip_range = config.get('RTSP', 'ip_range')
port = config.getint('RTSP', 'port', fallback=6001)
user = config.get('RTSP', 'user', fallback='admin')
password = config.get('RTSP', 'password', fallback='admin666')
subtype = config.getint('RTSP', 'subtype', fallback=1)
root_directory = config.get('RTSP', 'root_directory', fallback='E:/fmu/temp')
interval = config.getint('RTSP', 'interval', fallback=30)
ip_timeout = config.getint('RTSP', 'ip_timeout', fallback=10)
max_workers = config.getint('RTSP', 'max_workers', fallback=10)

# 目标检测配置
model_path = config.get('Detection', 'model_path', fallback='yolov8.onnx')
confidence_thres = config.getfloat('Detection', 'confidence_thres', fallback=0.5)
iou_thres = config.getfloat('Detection', 'iou_thres', fallback=0.5)
output_json = config.get('Detection', 'output_json', fallback='detection_results.json')

# ---------------------------- 全局资源 ----------------------------
# 创建线程池（避免在函数内部重复创建）
capture_executor = ThreadPoolExecutor(max_workers=max_workers)
detection_executor = ThreadPoolExecutor(max_workers=max_workers)


# ---------------------------- 工具函数 ----------------------------
def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def is_daytime():
    current_hour = time.localtime().tm_hour
    return 6 <= current_hour < 18


def get_ip_list(ip_range):
    start_ip, end_ip = ip_range.split('-')
    start_ip = ipaddress.IPv4Address(start_ip.strip())
    end_ip = ipaddress.IPv4Address(end_ip.strip())
    return [str(start_ip + i) for i in range(int(end_ip) - int(start_ip) + 1)]


# ---------------------------- 核心逻辑 ----------------------------
def capture_image(ip, channel, timestamp):
    rtsp_url = f"rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"
    logging.info(f"Connecting to {rtsp_url}")

    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logging.error(f"Failed to open RTSP stream: {ip} channel {channel}")
            return None

        ret, frame = cap.read()
        if not ret:
            logging.error(f"Failed to capture frame: {ip} channel {channel}")
            return None

        filename = f'{root_directory}/{ip}/{timestamp}/channel_{channel}.jpg'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, frame)
        logging.info(f"Saved image: {filename}")
        return filename

    except Exception as e:
        logging.error(f"Error capturing {ip} channel {channel}: {str(e)}")
        return None
    finally:
        if 'cap' in locals():
            cap.release()


def capture_from_ip(ip, timestamp):
    try:
        channels = [1, 3] if is_daytime() else [2, 3]
        image_paths = []
        for channel in channels:
            path = capture_image(ip, channel, timestamp)
            if path:
                image_paths.append(path)
        return ip, image_paths
    except Exception as e:
        logging.error(f"Error in capture_from_ip({ip}): {str(e)}")
        return ip, []


def async_process_images(model_path, img_dir, confidence_thres, iou_thres, output_json):
    return detection_executor.submit(
        process_images, model_path, img_dir, confidence_thres, iou_thres, output_json, logging
    )


# ---------------------------- 主函数 ----------------------------
def main():
    ip_list = get_ip_list(ip_range)
    os.makedirs(root_directory, exist_ok=True)

    try:
        while True:
            cycle_start = time.time()
            timestamp = get_timestamp()

            # ---------------------------- 步骤1: 并行截图并立即提交检测 ----------------------------
            capture_futures = []
            detection_futures = []  # 用于收集检测任务的future

            # 提交所有截图任务
            for ip in ip_list:
                future = capture_executor.submit(capture_from_ip, ip, timestamp)
                capture_futures.append(future)

            for future in concurrent.futures.as_completed(capture_futures, timeout=ip_timeout * len(ip_list)):
                try:
                    ip, paths = future.result(timeout=ip_timeout)
                except TimeoutError:
                    logging.warning("A capture task timed out")
                except Exception as e:
                    logging.error(f"Capture task error: {str(e)}")

            # 处理完成的截图任务并立即提交检测
            for future in concurrent.futures.as_completed(capture_futures, timeout=ip_timeout * len(ip_list)):
                try:
                    ip, paths = future.result(timeout=ip_timeout)
                    if paths:
                        # 立即提交检测任务
                        img_dir = os.path.dirname(paths[0])
                        detection_future = async_process_images(
                            model_path, img_dir,
                            confidence_thres, iou_thres, output_json
                        )
                        detection_futures.append(detection_future)
                        logging.info(f"Submitted detection for {ip} at {img_dir}")
                except TimeoutError:
                    logging.warning("A capture task timed out")
                except Exception as e:
                    logging.error(f"Capture task error: {str(e)}")

            # ---------------------------- 步骤2: 等待所有检测任务完成 ----------------------------
            for future in concurrent.futures.as_completed(detection_futures):
                try:
                    future.result(timeout=3600)
                except TimeoutError:
                    logging.error("Detection task timed out")
                except Exception as e:
                    logging.error(f"Detection task error: {str(e)}")

            # ---------------------------- 间隔控制 ----------------------------
            elapsed = time.time() - cycle_start
            sleep_time = max(0, interval - elapsed)
            logging.info(f"Cycle completed in {elapsed:.2f}s, sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("Received KeyboardInterrupt, shutting down...")
    finally:
        capture_executor.shutdown(wait=True)
        detection_executor.shutdown(wait=True)
        logging.info("All thread pools closed")


if __name__ == "__main__":
    main()