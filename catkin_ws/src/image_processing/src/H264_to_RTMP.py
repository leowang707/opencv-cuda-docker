import rospy
import numpy as np
import subprocess
from sensor_msgs.msg import CompressedImage

class RTMPStreamer:
    def __init__(self):
        rospy.init_node('rtmp_streamer', anonymous=True)
        self.image_sub = rospy.Subscriber("/camera_stitched/color/image_raw/avc", CompressedImage, self.image_callback, queue_size=1)
        
        # 設定 FFmpeg 推流到 RTMP 伺服器
        self.ffmpeg_process = subprocess.Popen(
            [
                'ffmpeg', '-re',
                '-fflags', 'nobuffer',  # 避免內部緩衝
                '-flags', 'low_delay',  # 低延遲
                '-f', 'h264',  # 確保是 H.264 裸流
                '-i', '-',
                '-c:v', 'copy',
                '-f', 'flv', 'rtmp://192.168.0.133/live/stream' #'rtmp://localhost/live/stream' # RTMP 伺服器位址
            ],
            stdin=subprocess.PIPE
        )

    def image_callback(self, msg):
        try:
            if self.ffmpeg_process.poll() is not None:
                rospy.logerr("FFmpeg 已經崩潰，重新啟動...")
                self.__init__()
            self.ffmpeg_process.stdin.write(msg.data)
            self.ffmpeg_process.stdin.flush()  # 確保立即發送
        except Exception as e:
            rospy.logerr(f"FFmpeg error: {e}")

    # def image_callback(self, msg):
    #     try:
    #         # 直接讀取 ROS topic 的 H.264 裸流
    #         self.ffmpeg_process.stdin.write(msg.data)
    #     except Exception as e:
    #         rospy.logerr(f"FFmpeg error: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = RTMPStreamer()
        node.run()
    except rospy.ROSInterruptException:
        pass