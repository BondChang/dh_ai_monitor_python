import cv2
import time
import os

# 登录信息
ip = "192.168.5.22"
port = 554  # RTSP默认端口
user = "admin"
password = "sanyou666"
subtype = 1  # 辅码流，节省流量

# 指定根目录
root_directory = "D:/temp/temp3"  # 保存图片的根目录

# 确保根目录存在
if not os.path.exists(root_directory):
    os.makedirs(root_directory)

# 获取当前时间戳
def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

# 获取当前时间的小时，判断是白天还是晚上
def is_daytime():
    current_hour = time.localtime().tm_hour
    # 假设白天为 6:00 - 18:00
    return 6 <= current_hour < 18

# 截取图片的函数
def capture_image(channel):
    # 构造RTSP URL
    rtsp_url = f"rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"
    print(f"Connecting to: {rtsp_url}")

    # 打开RTSP流
    cap = cv2.VideoCapture(rtsp_url)

    # 设置较低分辨率和较低帧率（例如 320x240 分辨率，5 FPS）
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    # cap.set(cv2.CAP_PROP_FPS, 5)  # 降低帧率

    # 检查是否成功连接
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream for channel {channel}.")
        return

    # 读取一帧
    ret, frame = cap.read()

    if ret:
        # 获取当前时间戳
        timestamp = get_timestamp()

        # 构造保存路径，文件名包括通道号和时间戳
        filename = f'{root_directory}/captured_image_channel_{channel}_{timestamp}.jpg'

        # 保存图片到指定目录
        cv2.imwrite(filename, frame)
        print(f"Image captured and saved as '{filename}'")
    else:
        print(f"Error: Could not capture frame from channel {channel}.")

    # 释放资源
    cap.release()

# 每30秒截取一次图片
while True:
    # 根据时间判断使用哪个通道
    if is_daytime():
        capture_image(1)  # 白天使用通道1
        capture_image(3)  # 白天使用通道1
    else:
        capture_image(2)  # 晚上使用通道2
        capture_image(3)  # 白天使用通道1

    # 每次截取后等待30秒
    time.sleep(60)
