#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import subprocess
import os
import fcntl
import select
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class HEVCRelay:
    """
    这个程序负责将 /camera_stitched/color/image_raw/compressed (JPEG) 转换成 H.265 裸流 (HEVC)
    并以 CompressedImage (format="hevc") 的形式发布到 /camera_stitched/color/image_raw/hevc。
    """
    def __init__(self):
        rospy.init_node('hevc_relay', anonymous=True)
        self.bridge = CvBridge()
        # 订阅来源影像 (JPEG)
        self.image_sub = rospy.Subscriber(
            "/camera_stitched/color/image_raw/compressed",
            CompressedImage, self.image_callback, queue_size=1
        )
        # 发布 HEVC 影像 (H.265)
        self.image_pub = rospy.Publisher(
            "/camera_stitched/color/image_raw/hevc",
            CompressedImage, queue_size=1
        )

        self.ffmpeg_process = None
        self.width = None
        self.height = None

    def start_ffmpeg(self, w, h):
        # 若之前有 ffmpeg 进程，就关闭
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.stdout.close()
            self.ffmpeg_process.stderr.close()
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()

        rospy.loginfo(f"Starting FFmpeg with resolution: {w}x{h}")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}',
            '-r', '6',   # 依需求更改FPS
            '-i', '-',
            '-c:v', 'libx265',  # 使用 H.265 编码器
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', '1',
            '-vf', 'format=yuv420p',  # 确保输出是 yuv420
            '-f', 'hevc',              # 输出为 H.265 HEVC 裸流
            '-'
        ]

        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        # 设置 stdout 和 stderr 为非阻塞模式
        fcntl.fcntl(self.ffmpeg_process.stdout, fcntl.F_SETFL, os.O_NONBLOCK)
        fcntl.fcntl(self.ffmpeg_process.stderr, fcntl.F_SETFL, os.O_NONBLOCK)

        self.width = w
        self.height = h

    def image_callback(self, msg):
        try:
            # 转换 ROS CompressedImage 为 OpenCV 图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                rospy.logerr("Failed to decode the image!")
                return

            # 获取影像尺寸
            h, w, _ = cv_image.shape
            rospy.loginfo(f"[HEVCRelay] Received image size: {w}x{h}")

            # 若解析度改变，重启 FFmpeg
            if self.width != w or self.height != h or self.ffmpeg_process is None:
                self.start_ffmpeg(w, h)

            # 将影像写入 ffmpeg
            try:
                self.ffmpeg_process.stdin.write(cv_image.tobytes())
            except Exception as e:
                rospy.logerr(f"Error writing to ffmpeg: {e}")
                return

            # 读取 ffmpeg 错误信息（非阻塞）
            try:
                err_msg = self.ffmpeg_process.stderr.read()
                if err_msg:
                    rospy.logwarn(f"FFmpeg error: {err_msg.decode(errors='ignore')}")
            except:
                pass

            # 读取 ffmpeg stdout（H.265 裸流，非阻塞）
            hevc_data = None
            rlist, _, _ = select.select([self.ffmpeg_process.stdout], [], [], 0)
            if self.ffmpeg_process.stdout in rlist:
                hevc_data = self.ffmpeg_process.stdout.read()

            if not hevc_data:
                rospy.logwarn("No valid H.265 data received from FFmpeg.")
                return

            # 发布 H.265 影像作为 ROS CompressedImage
            compressed_msg = CompressedImage()
            compressed_msg.header = msg.header
            compressed_msg.format = "hevc"  # 自定义压缩格式tag
            compressed_msg.data = hevc_data
            self.image_pub.publish(compressed_msg)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = HEVCRelay()
        node.run()
    except rospy.ROSInterruptException:
        pass
