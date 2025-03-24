import rospy
import cv2
import numpy as np
import ffmpeg
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from cv_bridge import CvBridge
import subprocess
import os
import fcntl
import select
# import socket

class AVCRelay:
    """
    這個程式負責將 /camera_stitched/color/image_raw/compressed (JPEG) 轉換成 H.264 (FLV格式)
    並直接推送到 RTMP 伺服器，以供即時串流播放。
    """
    def __init__(self):
        rospy.init_node('avc_relay', anonymous=True)
        self.bridge = CvBridge()
        
        # 訂閱來自 ROS 的影像 topic (JPEG)
        self.image_sub = rospy.Subscriber(
            "/camera_stitched/color/image_raw/compressed",
            CompressedImage, self.image_callback, queue_size=1
        )

        # # 設定 RTMP 伺服器地址
        self.rtmp_url = "rtmp://192.168.0.108/live/stream"  # 這裡請改為你的 RTMP 伺服器 IP
        
        self.ffmpeg_process = None
        self.width = None
        self.height = None

    def start_ffmpeg(self, w, h):
        # 若之前有 ffmpeg process，就關閉
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.stdout.close()
            self.ffmpeg_process.stderr.close()
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()

        rospy.loginfo(f"Starting FFmpeg with resolution: {w}x{h}, streaming to {self.rtmp_url}")
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}',
            '-r', '7',                  # 如有需要，請根據來源 FPS 調整
            '-i', '-',
            '-c:v', 'h264_nvenc',       # 使用 Nvidia GPU 硬體編碼
            '-preset', 'llhp',          # low latency high performance
            '-tune', 'll',              # 針對低延遲進行調整
            '-g', '7',                # GOP 大小設定 (每 6 幀一個 I-frame)
            '-keyint_min', '7',       # 最小關鍵影格間隔與 GOP 相同
            '-b:v', '1000k',          # 限制碼率，可依需求調整
            '-rc-lookahead', '0',     # 關閉 lookahead，降低延遲
            '-bf', '0',               # 關閉 B-frame
            '-probesize', '32',
            '-analyzeduration', '0',  # 加速初始化
            '-vf', 'format=yuv420p',  # 確保像素格式兼容性
            '-f', 'flv',
            self.rtmp_url             # 直接推送至 RTMP 伺服器
        ]

        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        # 設定 stdout 和 stderr 為非阻塞模式
        fcntl.fcntl(self.ffmpeg_process.stdout, fcntl.F_SETFL, os.O_NONBLOCK)
        fcntl.fcntl(self.ffmpeg_process.stderr, fcntl.F_SETFL, os.O_NONBLOCK)

        self.width = w
        self.height = h

    def image_callback(self, msg):
        try:
            # 轉換 ROS CompressedImage 為 OpenCV 圖像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                rospy.logerr("Failed to decode the image!")
                return

            # 取得影像尺寸
            h, w, _ = cv_image.shape
            rospy.loginfo(f"[AVCRelay] Received image size: {w}x{h}")

            # 若解析度改變，重啟 FFmpeg
            if self.width != w or self.height != h or self.ffmpeg_process is None:
                self.start_ffmpeg(w, h)

            # 將影像寫入 ffmpeg
            try:
                self.ffmpeg_process.stdin.write(cv_image.tobytes())
            except Exception as e:
                rospy.logerr(f"Error writing to ffmpeg: {e}")
                return

            # 讀取 ffmpeg 錯誤訊息（非阻塞）
            try:
                err_msg = self.ffmpeg_process.stderr.read()
                if err_msg:
                    rospy.logwarn(f"FFmpeg error: {err_msg.decode(errors='ignore')}")
            except:
                pass

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = AVCRelay()
        node.run()
    except rospy.ROSInterruptException:
        pass
