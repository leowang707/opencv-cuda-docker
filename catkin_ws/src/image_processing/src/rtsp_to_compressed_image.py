#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import time
from sensor_msgs.msg import CompressedImage
from threading import Thread

class RTSPCameraPublisher:
    def __init__(self, name, rtsp_url, topic_name):
        self.name = name
        self.rtsp_url = rtsp_url
        self.topic_name = topic_name
        self.publisher = rospy.Publisher(topic_name, CompressedImage, queue_size=1)

        # self.pipeline = (
        #     f"rtspsrc location={rtsp_url} latency=50 ! "
        #     "decodebin ! "
        #     "videoconvert ! "
        #     "video/x-raw,format=BGR ! "
        #     "appsink drop=true max-buffers=1 sync=false"
        # )
        
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
            except Exception as e:
                rospy.logerr(f"[{self.name}] 發布過程出錯：{e}")
                time.sleep(0.1)

def main():
    rospy.init_node("multi_rtsp_to_compressed_node", anonymous=True)

    cameras = [
        {"name": "cam1", "url": "rtsp://admin:admin@192.168.0.101:554/video", "topic": "/camera1/color/image_raw/compressed"},
        {"name": "cam2", "url": "rtsp://admin:admin@192.168.0.102:554/video", "topic": "/camera2/color/image_raw/compressed"},
        {"name": "cam3", "url": "rtsp://admin:admin@192.168.0.103:554/video", "topic": "/camera3/color/image_raw/compressed"},
        # {"name": "cam4", "url": "rtsp://admin:admin@192.168.131.104:554/video", "topic": "/camera4/color/image_raw/compressed"},
    ]

    nodes = []
    for cam in cameras:
        node = RTSPCameraPublisher(cam["name"], cam["url"], cam["topic"])
        node.start()
        nodes.append(node)

    rospy.loginfo("✅ 所有 RTSP 相機串流節點啟動完成。")
    rospy.spin()

    for node in nodes:
        node.stop()

if __name__ == "__main__":
    main()
