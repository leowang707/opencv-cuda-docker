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

class AVCRelay:
    """
    這個程式負責將 /camera_stitched/color/image_raw/compressed (JPEG) 轉換成 H.264 裸流 (AVC)
    並以 CompressedImage (format="avc") 的形式發佈到 /camera_stitched/color/image_raw/avc。

    與原本的 h264 或其他格式類似，只是將 ffmpeg 最終輸出格式改為 "avc"，
    你也可以調整編碼參數 (bitrate、fps、解析度)，提高或降低壓縮效果。
    """
    def __init__(self):
        rospy.init_node('avc_relay', anonymous=True)
        self.bridge = CvBridge()
        # 訂閱來源影像 (JPEG)
        self.image_sub = rospy.Subscriber(
            "/camera_stitched1/color/image_raw/compressed",
            CompressedImage, self.image_callback, queue_size=1
        )
        # 發布 AVC 影像 (H.264)
        self.image_pub = rospy.Publisher(
            "/camera_stitched/color/image_raw/avc",
            CompressedImage, queue_size=1
        )

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

        rospy.loginfo(f"Starting FFmpeg with resolution: {w}x{h}")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}',
            '-r', '6',   # FPS 設定為 6，可根據需求調整
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', '6',  # 調整 GOP 為 30（避免 slice header error）
            '-threads', '8',  # 使用 8 個 CPU 核心
            '-keyint_min', '15',  # 設定最小關鍵影格間隔，平衡影像品質與流暢度
            '-x264-params', 'slice-max-size=500',  # 限制 slice 大小，減少解碼錯誤
            '-bufsize', '500k',  # 增加緩衝，降低掉幀
            '-b:v', '1000k',  # 限制碼率，避免編碼異常
            '-vf', 'format=yuv420p',
            '-f', 'h264',
            '-'
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

            # 讀取 ffmpeg stdout（H.264 裸流，非阻塞）
            avc_data = None
            rlist, _, _ = select.select([self.ffmpeg_process.stdout], [], [], 0)
            if self.ffmpeg_process.stdout in rlist:
                avc_data = self.ffmpeg_process.stdout.read()

            if not avc_data:
                rospy.logwarn("No valid H.264 data received from FFmpeg.")
                return

            # 發布 H.264 影像作為 ROS CompressedImage
            compressed_msg = CompressedImage()
            compressed_msg.header = msg.header
            compressed_msg.format = "avc"  # 自定義壓縮格式tag
            compressed_msg.data = avc_data
            self.image_pub.publish(compressed_msg)

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
