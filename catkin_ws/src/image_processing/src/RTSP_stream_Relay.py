#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import subprocess
import os
import fcntl
import time

class AVCRtspRelay:
    def __init__(self):
        rospy.init_node('avc_rtsp_relay', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/camera_stitched/color/image_raw/compressed",
            CompressedImage, self.image_callback, queue_size=1
        )

        self.rtsp_url = "rtsp://192.168.0.165:8554/mystream"
        self.ffmpeg_process = None
        self.width = None
        self.height = None
        self.last_restart_time = 0
        self.retry_interval = 5  # 秒

        self.fixed_fps = 6  # 固定推流偵率

    def stop_ffmpeg(self):
        if self.ffmpeg_process:
            rospy.logwarn("🔻 關閉 FFmpeg")
            try:
                self.ffmpeg_process.stdin.close()
            except:
                pass
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except:
                pass
            self.ffmpeg_process = None

    def start_ffmpeg(self, w, h):
        now = time.time()
        if now - self.last_restart_time < self.retry_interval:
            return  # 尚未到達重試時間

        self.last_restart_time = now
        self.stop_ffmpeg()

        rospy.loginfo(f"🚀 啟動 FFmpeg：解析度 {w}x{h}, FPS={self.fixed_fps}, 推送到 {self.rtsp_url}")

        cmd = [
            'ffmpeg', '-re',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}',
            '-r', str(self.fixed_fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-g', '10',
            '-keyint_min', '10',
            '-b:v', '1500k',
            '-bf', '0',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            self.rtsp_url
        ]

        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            fcntl.fcntl(self.ffmpeg_process.stderr, fcntl.F_SETFL, os.O_NONBLOCK)
            self.width = w
            self.height = h
        except Exception as e:
            rospy.logerr(f"❌ FFmpeg 啟動失敗: {e}")
            self.ffmpeg_process = None

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                rospy.logerr("❌ 影像解碼失敗")
                return

            h, w, _ = cv_image.shape

            if self.ffmpeg_process is None or self.width != w or self.height != h:
                self.start_ffmpeg(w, h)

            if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
                rospy.logwarn("⚠️ FFmpeg 進程已結束，略過這幀")
                self.stop_ffmpeg()
                return

            try:
                self.ffmpeg_process.stdin.write(cv_image.tobytes())
            except BrokenPipeError:
                rospy.logwarn("📛 FFmpeg Broken pipe，正在重啟...")
                self.stop_ffmpeg()
                return
            except Exception as e:
                rospy.logwarn(f"📛 FFmpeg 寫入錯誤：{e}")
                self.stop_ffmpeg()
                return

            # 非阻塞讀 stderr，檢查錯誤
            try:
                err = self.ffmpeg_process.stderr.read()
                if err:
                    decoded = err.decode(errors='ignore')
                    if "Connection refused" in decoded or "not found" in decoded:
                        rospy.logwarn(f"[FFmpeg 錯誤] {decoded.strip()}")
                        self.stop_ffmpeg()
                    else:
                        rospy.loginfo_throttle(5, f"[FFmpeg] {decoded.strip()}")
            except:
                pass

        except Exception as e:
            rospy.logerr(f"[Callback Error] {e}")

    def run(self):
        rospy.spin()
        self.stop_ffmpeg()

if __name__ == '__main__':
    try:
        node = AVCRtspRelay()
        node.run()
    except rospy.ROSInterruptException:
        pass
