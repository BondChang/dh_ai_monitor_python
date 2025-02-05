import cv2
import time
import os
import configparser
import ipaddress
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from ai_detect import process_images

# 配置日志
log_file = 'capture.log'
max_log_size = 5 * 1024 * 1024  # 5MB
backup_count = 500

handler = RotatingFileHandler(
    log_file, maxBytes=max_log_size, backupCount=backup_count
)

logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
excluded_ips = config.get('RTSP', 'excluded_ips', fallback='').split(',')
# 目标检测配置
model_path = config.get('Detection', 'model_path', fallback='yolov8.onnx')
confidence_thres = config.getfloat('Detection', 'confidence_thres', fallback=0.5)
iou_thres = config.getfloat('Detection', 'iou_thres', fallback=0.5)
output_json = config.get('Detection', 'output_json', fallback='detection_results.json')

detection_executor = ThreadPoolExecutor(max_workers=max_workers)
# 确保根目录存在
os.makedirs(root_directory, exist_ok=True)


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def is_daytime():
    current_hour = time.localtime().tm_hour
    return 6 <= current_hour < 18


def safe_capture(ip, channel, timestamp, timeout):
    """带超时控制的截图函数"""
    start_time = time.time()
    rtsp_url = f"rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}&rtsp_transport=tcp"

    try:
        # 创建带超时参数的VideoCapture
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.setExceptionMode(True)
        # cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)  # 设置连接超时

        # 第一阶段：连接摄像头
        connect_start = time.time()
        while not cap.isOpened():
            if time.time() - connect_start > timeout:
                logging.error(f"[{ip}] 连接超时 ({timeout}s)")
                return False
            time.sleep(0.1)  # 避免CPU空转

        # 第二阶段：读取视频帧
        read_start = time.time()
        while True:
            # 检查总超时时间
            if time.time() - start_time > timeout:
                logging.error(f"[{ip}] 抓取超时 ({timeout}s)")
                return False

            # 尝试抓取帧
            grabbed = cap.grab()  # 更高效的抓取方式
            if grabbed:
                ret, frame = cap.retrieve()
                if ret:
                    # 保存文件
                    save_path = f"{root_directory}/{ip}/{timestamp}/channel_{channel}.jpg"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, frame)
                    logging.info(f"[{ip}] 成功保存到 {save_path}")
                    return True

            # 渐进式等待策略
            elapsed = time.time() - read_start
            if elapsed > timeout / 2:  # 后半段增加检查频率
                time.sleep(0.1)
            else:
                time.sleep(0.5)

    except Exception as e:
        logging.error(f"[{ip}] 捕获异常: {str(e)}")
        return False
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()


def process_ip(ip):
    """处理单个IP地址"""
    timestamp = get_timestamp()
    channels = [1, 3] if is_daytime() else [2, 3]

    total_success = 0
    start_time = time.time()

    for channel in channels:
        remaining_time = ip_timeout - (time.time() - start_time)
        if remaining_time <= 0:
            logging.warning(f"[{ip}] 通道{channel}因超时跳过")
            break

        if safe_capture(ip, channel, timestamp, remaining_time):
            total_success += 1

    # 返回成功数和时间戳
    return total_success, timestamp  # 修改返回值


# 获取IP范围并生成IP列表
def get_ip_list(ip_range):
    start_ip, end_ip = ip_range.split('-')
    start_ip = ipaddress.IPv4Address(start_ip.strip())
    end_ip = ipaddress.IPv4Address(end_ip.strip())

    ip_list = []
    current_ip = start_ip

    # 从start_ip到end_ip遍历，生成所有IP
    while current_ip <= end_ip:
        ip_str = str(current_ip)
        # 如果当前IP在排除列表中，则跳过
        if ip_str not in excluded_ips:
            ip_list.append(ip_str)
        current_ip += 1  # 下一地址

    return ip_list


def async_process_images(model_path, img_dir, confidence_thres, iou_thres, output_json):
    return detection_executor.submit(
        process_images, model_path, img_dir, confidence_thres, iou_thres, output_json, logging
    )


def main():
    ip_list = get_ip_list(ip_range)

    while True:
        cycle_start = time.time()
        success_count = 0

        for ip in ip_list:
            logging.info(f"开始处理IP: {ip}")
            start_ip = time.time()

            try:
                # 获取新的返回值
                success, timestamp = process_ip(ip)  # 修改这里
                success_count += success

                # 如果有成功抓取则提交分析任务
                if success > 0:
                    img_dir = os.path.join(root_directory, ip, timestamp)

                    # 提交异步分析任务
                    future = async_process_images(
                        model_path,
                        img_dir,
                        confidence_thres,
                        iou_thres,
                        output_json
                    )

                    # 添加结果回调（可选）
                    future.add_done_callback(
                        lambda f: logging.info(f"ai分析完成: {img_dir}")
                        if not f.exception()
                        else logging.error(f"ai分析失败: {img_dir} - {f.exception()}")
                    )

            except Exception as e:
                logging.error(f"[{ip}] 处理失败: {str(e)}")

            # 记录IP处理时长
            ip_duration = time.time() - start_ip
            logging.info(f"[{ip}] 处理完成，耗时 {ip_duration:.2f}s")
        # 周期控制（保持原样）
        cycle_duration = time.time() - cycle_start
        if cycle_duration < 925:
            time.sleep(925 - cycle_duration)
        logging.info(f"本轮完成，成功率: {success_count}/{len(ip_list) * 2}，总耗时: {cycle_duration:.2f}s")

        if cycle_duration < interval:
            sleep_time = interval - cycle_duration
            logging.info(f"等待下次循环: {sleep_time:.2f}s")
            time.sleep(sleep_time)
        else:
            logging.warning("警告：循环周期超过设定间隔")


if __name__ == "__main__":
    main()
