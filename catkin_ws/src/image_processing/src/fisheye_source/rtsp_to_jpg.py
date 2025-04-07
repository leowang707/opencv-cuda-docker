#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import time
import os
import threading
from sensor_msgs.msg import CompressedImage
from threading import Thread
import rospkg

# 根據 choice 選擇相機
choice = 2
if choice == 3:
    side = "left"
elif choice == 2:
    side = "mid"
elif choice == 1:
    side = "right"
else:
    side = "back"

# 定義單台相機的 RTSP URL 和對應的 ROS topic
if side == "left":
    camera_config = {
        "url": "rtsp://192.168.0.101:554/video",
        "topic": "/camera3/color/image_raw/compressed"
    }
elif side == "mid":
    camera_config = {
        "url": "rtsp://192.168.0.102:554/video",
        "topic": "/camera2/color/image_raw/compressed"
    }
elif side == "right":
    camera_config = {
        "url": "rtsp://192.168.0.103:554/video",
        "topic": "/camera1/color/image_raw/compressed"
    }
elif side == "back":
    camera_config = {
        "url": "rtsp://192.168.0.104:554/video",
        "topic": "/camera4/color/image_raw/compressed"
    }

# 使用 rospkg 獲取 image_processing 包的路徑
rospack = rospkg.RosPack()
package_path = rospack.get_path('image_processing')  # 假設包名為 image_processing
# 設定儲存圖片的資料夾
SAVE_DIR = os.path.join(package_path, "src", "fisheye_source", f"camera_{side}")
os.makedirs(SAVE_DIR, exist_ok=True)

# 用於控制拍照的全局變數
take_picture = threading.Event()

class RTSP2JPG:
    def __init__(self, name, rtsp_url, topic_name, side):
        self.name = name
        self.rtsp_url = rtsp_url
        self.topic_name = topic_name
        self.side = side  # 用於標識相機位置
        self.publisher = rospy.Publisher(topic_name, CompressedImage, queue_size=1)

        self.pipeline = (
            f"rtspsrc location={rtsp_url} latency=0 protocols=udp drop-on-latency=true ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            rospy.logerr(f"[{self.name}] ❌ 無法開啟 RTSP 串流：{rtsp_url}")
            self.cap = None

        self.thread = Thread(target=self.stream_loop)
        self.thread.daemon = True
        self.running = True

    def start(self):
        if self.cap:
            self.thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def stream_loop(self):
        global take_picture, SAVE_DIR
        while not rospy.is_shutdown() and self.running:
            if self.cap is None:
                time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn(f"[{self.name}] ⚠️ 讀取 RTSP 影格失敗")
                time.sleep(0.1)
                continue

            try:
                msg = CompressedImage()
                msg.header.stamp = rospy.Time.now()
                msg.format = "jpeg"

                success, encoded_img = cv2.imencode(".jpg", frame)
                if not success:
                    rospy.logwarn(f"[{self.name}] ❌ OpenCV 影像壓縮失敗")
                    time.sleep(0.05)
                    continue

                msg.data = np.array(encoded_img).tobytes()
                self.publisher.publish(msg)

                # 等待拍照指令
                if take_picture.is_set():
                    # 儲存圖片
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{SAVE_DIR}/camera_{self.side}_{timestamp_str}.jpg"
                    cv2.imwrite(filename, frame)
                    rospy.loginfo(f"[{self.name}] 已儲存圖片: {filename}")
                    take_picture.clear()  # 重置拍照標誌

            except Exception as e:
                rospy.logerr(f"[{self.name}] 發布過程出錯：{e}")
                time.sleep(0.1)

def keyboard_listener():
    """監聽鍵盤輸入，按下 'y' 觸發拍照"""
    while not rospy.is_shutdown():
        key = input("按下 'y' 拍照 (或輸入 'q' 退出): ").lower()
        if key == 'y':
            take_picture.set()
        elif key == 'q':
            rospy.signal_shutdown("使用者退出")
            break

def main():
    global take_picture, SAVE_DIR
    rospy.init_node("single_rtsp_to_compressed_node", anonymous=True)

    # 啟動單台相機
    node = RTSP2JPG(f"camera_{side}", camera_config["url"], camera_config["topic"], side)
    node.start()

    # 啟動鍵盤監聽器線程
    keyboard_thread = Thread(target=keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()

    rospy.loginfo(f"✅ RTSP 相機 ({side}) 串流節點啟動完成。")
    rospy.loginfo("使用 'y' 拍照，'q' 退出")
    rospy.spin()

    node.stop()

if __name__ == "__main__":
    main()