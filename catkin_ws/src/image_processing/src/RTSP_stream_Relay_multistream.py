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

class FFmpegStreamer:
    def __init__(self, rtsp_url, fixed_fps=6, resolution=None):
        self.rtsp_url = rtsp_url
        self.fixed_fps = fixed_fps
        self.resolution = resolution  # (w, h) 或 None
        self.ffmpeg_process = None
        self.width = None
        self.height = None
        self.last_restart_time = 0
        self.retry_interval = 5

    def stop_ffmpeg(self):
        if self.ffmpeg_process:
            rospy.logwarn(f"🔻 關閉 FFmpeg：{self.rtsp_url}")
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
            return

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
            rospy.logerr(f"❌ FFmpeg 啟動失敗（{self.rtsp_url}）: {e}")
            self.ffmpeg_process = None

    def push_frame(self, cv_image):
        if self.resolution:
            cv_image = cv2.resize(cv_image, self.resolution)
        h, w, _ = cv_image.shape

        if self.ffmpeg_process is None or self.width != w or self.height != h:
            self.start_ffmpeg(w, h)

        if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
            rospy.logwarn(f"⚠️ FFmpeg 進程已結束（{self.rtsp_url}），略過這幀")
            self.stop_ffmpeg()
            return

        try:
            self.ffmpeg_process.stdin.write(cv_image.tobytes())
        except BrokenPipeError:
            rospy.logwarn(f"📛 FFmpeg Broken pipe（{self.rtsp_url}），正在重啟...")
            self.stop_ffmpeg()
        except Exception as e:
            rospy.logwarn(f"📛 FFmpeg 寫入錯誤（{self.rtsp_url}）：{e}")
            self.stop_ffmpeg()

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


class AVCMultiRtspRelay:
    def __init__(self):
        rospy.init_node('avc_multi_rtsp_relay', anonymous=True)
        self.bridge = CvBridge()

        # 支援每路的 URL、FPS 與解析度設定
        self.streams = {
            "/camera_stitched/color/image_raw/compressed": {
                "url": "rtsp://192.168.0.108:8554/mystream",
                "fps": 60,
                "resolution": (1920, 480)
            },
            "/halo_radar/radar_image/compressed": {
                "url": "rtsp://192.168.0.108:8554/radar",
                "fps": 1,
                "resolution": (480, 480)  # 使用原始大小
            }
        }

        self.stream_objs = {}

        for topic, cfg in self.streams.items():
            streamer = FFmpegStreamer(cfg["url"], cfg["fps"], cfg["resolution"])
            self.stream_objs[topic] = streamer
            rospy.Subscriber(topic, CompressedImage, self.make_callback(streamer), queue_size=1)

    def make_callback(self, streamer):
        def callback(msg):
            try:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv_image is not None:
                    streamer.push_frame(cv_image)
                else:
                    rospy.logerr("❌ 解碼失敗")
            except Exception as e:
                rospy.logerr(f"[Callback Error] {e}")
        return callback

    def run(self):
        rospy.spin()
        for streamer in self.stream_objs.values():
            streamer.stop_ffmpeg()


if __name__ == '__main__':
    try:
        node = AVCMultiRtspRelay()
        node.run()
    except rospy.ROSInterruptException:
        pass
