import cv2
import time
import os
import threading
import configparser
import ipaddress
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import logging
from logging.handlers import RotatingFileHandler

# 配置日志
log_file = 'capture.log'  # 日志文件路径
max_log_size = 5 * 1024 * 1024  # 最大日志文件大小 10MB
backup_count = 500

# 创建 RotatingFileHandler
handler = RotatingFileHandler(
    log_file, maxBytes=max_log_size, backupCount=backup_count
)

# 配置日志
logging.basicConfig(
    handlers=[handler],  # 使用 RotatingFileHandler
    level=logging.INFO,  # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 日期格式
)

# 从配置文件加载参数
config = configparser.ConfigParser()
config.read('config.ini')

# 配置文件中的参数
ip_range = config.get('RTSP', 'ip_range')  # IP范围，格式为 192.168.5.1-192.168.5.68
port = config.getint('RTSP', 'port', fallback=6001)
user = config.get('RTSP', 'user', fallback='admin')
password = config.get('RTSP', 'password', fallback='admin666')
subtype = config.getint('RTSP', 'subtype', fallback=1)  # 辅码流
root_directory = config.get('RTSP', 'root_directory', fallback='E:/fmu/temp')
interval = config.getint('RTSP', 'interval', fallback=30)  # 时间间隔，单位秒
ip_timeout = config.getint('RTSP', 'ip_timeout', fallback=10)  # 单个IP超时时间，单位秒
max_workers = config.getint('RTSP', 'max_workers', fallback=1)  # 最大工作线程数

# 确保根目录存在
if not os.path.exists(root_directory):
    os.makedirs(root_directory)


# 获取当前时间戳
def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


# 获取当前时间的小时，判断是白天还是晚上
def is_daytime():
    current_hour = time.localtime().tm_hour
    return 6 <= current_hour < 18


# 截取图片的函数
def capture_image(ip, channel, timestamp):
    start_time = time.time()  # 记录开始时间

    # 构造RTSP URL
    rtsp_url = f"rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"
    logging.info(f"Connecting to: {rtsp_url}")

    # 打开RTSP流
    cap = cv2.VideoCapture(rtsp_url)

    # 检查是否成功连接
    if not cap.isOpened():
        logging.error(f"Error: Could not open RTSP stream for IP {ip}, channel {channel}.")
        elapsed_time = time.time() - start_time
        return elapsed_time

    # 读取一帧
    ret, frame = cap.read()

    if ret:
        # 构造保存路径，文件名包括IP地址、通道号和时间戳
        filename = f'{root_directory}/{ip}/{timestamp}/channel_{channel}.jpg'

        # 确保文件夹存在
        folder_path = os.path.dirname(filename)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存图片到指定目录
        cv2.imwrite(filename, frame)
        logging.info(f"Image captured and saved as '{filename}'")
    else:
        logging.error(f"Error: Could not capture frame from IP {ip}, channel {channel}.")

    # 释放资源
    cap.release()

    # 返回本次截图操作的耗时
    elapsed_time = time.time() - start_time
    return elapsed_time


# 获取IP范围并生成IP列表
def get_ip_list(ip_range):
    start_ip, end_ip = ip_range.split('-')
    start_ip = ipaddress.IPv4Address(start_ip.strip())
    end_ip = ipaddress.IPv4Address(end_ip.strip())

    ip_list = []
    current_ip = start_ip

    # 从start_ip到end_ip遍历，生成所有IP
    while current_ip <= end_ip:
        ip_list.append(str(current_ip))
        current_ip += 1  # 下一地址

    return ip_list


# 捕获图片的线程目标函数，包含超时处理
def capture_from_ip(ip, timestamp):
    try:
        # 截取当前IP的图片
        capture_times = []
        if is_daytime():
            capture_times.append(capture_image(ip, 1, timestamp))  # 白天使用通道1
            capture_times.append(capture_image(ip, 3, timestamp))  # 白天使用通道3
        else:
            capture_times.append(capture_image(ip, 2, timestamp))  # 晚上使用通道2
            capture_times.append(capture_image(ip, 3, timestamp))  # 晚上使用通道3

        # 计算当前IP截图操作的总耗时
        total_time = sum(time for time in capture_times if time is not None)  # 总耗时

        return total_time
    except Exception as e:
        logging.error(f"Exception occurred while capturing from IP {ip}: {e}")
        return None


# 主函数，遍历所有IP并控制截图间隔
def main():
    ip_list = get_ip_list(ip_range)
    while True:
        start_time = time.time()  # 记录开始时间
        total_capture_time = 0

        # 使用线程池并发处理每个IP
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ip = {executor.submit(capture_from_ip, ip, get_timestamp()): ip for ip in ip_list}

            # 处理每个IP的结果
            for future in future_to_ip:
                try:
                    ip_capture_time = future.result(timeout=ip_timeout)
                    if ip_capture_time is not None:
                        total_capture_time += ip_capture_time
                    else:
                        logging.warning(f"Skipping IP {future_to_ip[future]} due to error or timeout.")
                except TimeoutError:
                    logging.warning(f"Timeout while capturing from IP {future_to_ip[future]}.")

        print('123')
        # 判断遍历的时间是否小于间隔时间
        elapsed_time = time.time() - start_time  # 本轮操作的耗时
        logging.info(f"Total capture time for this round: {elapsed_time:.2f} seconds")
        if elapsed_time < interval:
            logging.info(f"Waiting for {interval - elapsed_time:.2f} seconds to complete the interval.")
            time.sleep(interval - elapsed_time)  # 等待剩余时间
        else:
            logging.warning(f"Interval exceeded. Proceeding to the next round.")


# 运行主函数
if __name__ == "__main__":
    main()
