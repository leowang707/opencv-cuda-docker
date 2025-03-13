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
import socket

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
            "/camera_stitched1/color/image_raw/compressed",
            CompressedImage, self.image_callback, queue_size=1
        )
        
        # 自動偵測本機 IP
        local_ip = self.get_local_ip()
        rospy.loginfo(f"[AVCRelay] Local IP detected: {local_ip}")

        # 設定 RTMP 伺服器地址
        self.rtmp_url = f"rtmp://{local_ip}/live/stream"

        # # 設定 RTMP 伺服器地址
        # self.rtmp_url = "rtmp://192.168.0.166/live/stream"  # 這裡請改為你的 RTMP 伺服器 IP
        
        self.ffmpeg_process = None
        self.width = None
        self.height = None

    def get_local_ip(self):
        """
        利用 socket 連到一個外部位址(如 8.8.8.8)後，
        讀取連線的本端位址來取得本機對外可用的 IP。
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception as e:
            rospy.logwarn(f"Failed to get local IP via socket: {e}")
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    def start_ffmpeg(self, w, h):
        # 若之前有 ffmpeg process，就關閉
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.stdout.close()
            self.ffmpeg_process.stderr.close()
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()

        rospy.loginfo(f"Starting FFmpeg with resolution: {w}x{h}, streaming to {self.rtmp_url}")

        # FFmpeg 指令，將 JPEG 轉換為 FLV 並推送至 RTMP 伺服器
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}',
            '-r', '6',  # 設定 FPS 為 30，提高流暢度
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',  # 最快壓縮，減少計算負擔
            '-tune', 'zerolatency',  # 降低延遲，適合即時串流
            '-g', '6',  # 設定 GOP 為 30，提高壓縮效率
            '-keyint_min', '10',  # 設定最小關鍵影格間隔
            '-x264-params', 'slice-max-size=500',  # 避免 slice header error
            '-bufsize', '500k',  # 增加緩衝，降低掉幀
            '-b:v', '1000k',  # 限制碼率，提高畫質
            '-threads', '8',  # 使用 8 個 CPU 核心，提升效率
            '-rtbufsize', '200M',  # 減少緩衝延遲
            '-probesize', '32', '-analyzeduration', '0',  # 加速初始化
            '-vf', 'format=yuv420p',
            '-f', 'flv',  # 變更為 FLV 格式，適合 RTMP 推流
            self.rtmp_url  # 直接推送到 RTMP 伺服器
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
