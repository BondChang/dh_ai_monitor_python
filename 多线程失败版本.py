import cv2
import time
import os
import configparser
import ipaddress
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed, ThreadPoolExecutor
import threading

# 配置日志
log_file = 'capture.log'
max_log_size = 5 * 1024 * 1024
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

# 从配置文件加载参数
config = configparser.ConfigParser()
config.read('config.ini')

# 配置文件中的参数
ip_range = config.get('RTSP', 'ip_range')
port = config.getint('RTSP', 'port', fallback=554)
user = config.get('RTSP', 'user', fallback='admin')
password = config.get('RTSP', 'password', fallback='admin666')
subtype = config.getint('RTSP', 'subtype', fallback=0)
root_directory = config.get('RTSP', 'root_directory', fallback='E:/fmu/temp')
interval = config.getint('RTSP', 'interval', fallback=30)
ip_timeout = config.getint('RTSP', 'ip_timeout', fallback=10)  # 每个通道的超时时间
max_workers = config.getint('RTSP', 'max_workers', fallback=4)

if not os.path.exists(root_directory):
    os.makedirs(root_directory)


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def is_daytime():
    current_hour = time.localtime().tm_hour
    return 6 <= current_hour < 18


def capture_image_with_timeout(ip, channel, timestamp):
    """带超时控制的图像捕获函数"""

    def capture():
        try:
            rtsp_url = f"rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"
            logging.info(f"正在连接摄像头 {ip} 通道 {channel}")

            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                logging.error(f"摄像头 {ip} 通道 {channel} 连接失败")
                return None

            ret, frame = cap.read()
            logging.info(f"完成连接摄像头 {ip} 通道 {channel}")
            if ret:
                filename = f'{root_directory}/{ip}/{timestamp}/channel_{channel}.jpg'
                folder_path = os.path.dirname(filename)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                cv2.imwrite(filename, frame)
                logging.info(f"摄像头 {ip} 通道 {channel} 截图已保存: {filename}")
                return True
            else:
                logging.error(f"摄像头 {ip} 通道 {channel} 获取图像失败")
                return None
        except Exception as e:
            logging.error(f"摄像头 {ip} 通道 {channel} 发生异常: {str(e)}")
            return None
        finally:
            if 'cap' in locals():
                cap.release()

    # 使用线程实现超时控制
    result = [None]
    capture_thread = threading.Thread(target=lambda: result.__setitem__(0, capture()))
    capture_thread.daemon = True
    capture_thread.start()
    capture_thread.join(timeout=ip_timeout)

    if capture_thread.is_alive():
        logging.warning(f"摄像头 {ip} 通道 {channel} 处理超时")
        return None

    return result[0]



def get_ip_list(ip_range):
    start_ip, end_ip = ip_range.split('-')
    start_ip = ipaddress.IPv4Address(start_ip.strip())
    end_ip = ipaddress.IPv4Address(end_ip.strip())

    ip_list = []
    current_ip = start_ip
    while current_ip <= end_ip:
        ip_list.append(str(current_ip))
        current_ip += 1

    return ip_list


def process_single_ip(ip):
    """处理单个IP的所有通道"""
    timestamp = get_timestamp()
    success_count = 0
    channels = [1, 3] if is_daytime() else [2, 3]

    for channel in channels:
        result = capture_image_with_timeout(ip, channel, timestamp)
        if result:
            success_count += 1

    return ip, success_count


def main():
    ip_list = get_ip_list(ip_range)

    while True:
        start_time = time.time()
        total_success = 0
        total_fail = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_ip = {executor.submit(process_single_ip, ip): ip for ip in ip_list}

            # 收集结果
            for future in as_completed(future_to_ip):
                ip = future_to_ip[future]
                try:
                    result = future.result()
                    if result:
                        _, success_count = result
                        total_success += success_count
                        if success_count == 0:
                            total_fail += 1
                    else:
                        total_fail += 1
                except Exception as e:
                    logging.error(f"处理摄像头 {ip} 时发生异常: {str(e)}")
                    total_fail += 1

        elapsed_time = time.time() - start_time
        logging.info(f"""
        本轮统计:
        - 总耗时: {elapsed_time:.2f} 秒
        - 成功截图数: {total_success}
        - 失败IP数: {total_fail}
        """)

        if elapsed_time < interval:
            wait_time = interval - elapsed_time
            logging.info(f"等待 {wait_time:.2f} 秒进入下一轮截图")
            time.sleep(wait_time)
        else:
            logging.warning(f"本轮执行超时 {elapsed_time - interval:.2f} 秒")


if __name__ == "__main__":
    main()