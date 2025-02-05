import socket

import cv2
import time
import configparser
import ipaddress

# -------------------- 读取配置 --------------------
config = configparser.ConfigParser()
config.read('config.ini')

# RTSP 配置
ip_range = config.get('RTSP', 'ip_range')
port = config.getint('RTSP', 'port', fallback=6001)
user = config.get('RTSP', 'user', fallback='admin')
password = config.get('RTSP', 'password', fallback='admin666')
subtype = config.getint('RTSP', 'subtype', fallback=1)
ip_timeout = config.getint('RTSP', 'ip_timeout', fallback=10)

# -------------------- 工具函数 --------------------
def get_ip_list(ip_range):
    """
    根据配置的 ip_range（例如 "192.168.1.100-192.168.1.200"）生成 IP 列表
    """
    start_ip, end_ip = ip_range.split('-')
    start_ip = ipaddress.IPv4Address(start_ip.strip())
    end_ip = ipaddress.IPv4Address(end_ip.strip())

    ip_list = []
    while start_ip <= end_ip:
        ip_list.append(str(start_ip))
        start_ip += 1
    return ip_list


def check_rtsp_via_socket(ip, port, timeout=5):
    """
    使用 socket 检测 RTSP 连接是否连通
    参数:
      ip: 摄像头 IP 地址
      port: RTSP 服务端口（如6001）
      timeout: 超时时间（秒）
    返回:
      True 表示连接成功，False 表示连接失败
    """
    try:
        sock = socket.create_connection((ip, port), timeout)
        sock.close()
        return True
    except Exception as e:
        return False


def check_rtsp_connection(ip, channel, timeout=10):
    """
    检测指定 IP 与通道的 RTSP 连接是否连通，只判断能否建立连接
    参数:
      ip: 摄像头 IP 地址
      channel: 通道号
      timeout: 超时时间（秒）
    返回:
      True 表示连接成功；False 表示连接失败
    """
    rtsp_url = (
        f"rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?"
        f"channel={channel}&subtype={subtype}&rtsp_transport=tcp"
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    start_time = time.time()
    while not cap.isOpened():
        if time.time() - start_time > timeout:
            cap.release()
            return False
        time.sleep(0.1)
    cap.release()
    return True

def main():
    ip_list = get_ip_list(ip_range)
    problematic_ips = []
    channels = [1]


    for ip in ip_list:
        print(f"正在检测 {ip} 的 RTSP 连接...")
        overall_status = False
        for channel in channels:
            # 如果任意一个通道连通，则认为该 ip 整体连通
            if check_rtsp_connection(ip, channel, 5):
                overall_status = True
                break
        if not overall_status:
            print("连接失败")
            problematic_ips.append(ip)
        else:
            print("连接成功")

    print("\n检测结果：")
    if problematic_ips:
        print("以下 IP 存在问题：")
        for ip in problematic_ips:
            print(f"  {ip}")
    else:
        print("所有 IP 均连通！")

if __name__ == '__main__':
    main()
    input("Press Enter to exit...")
