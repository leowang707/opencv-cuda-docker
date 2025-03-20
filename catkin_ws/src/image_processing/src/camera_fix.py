#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class CameraUndistorter:
    def __init__(self, input_topic, output_topic, dim, K, D, balance=1.0):
        """
        初始化單一鏡頭的 undistorter。
        :param input_topic: 原始魚眼影像 topic，例如 "/camera1/color/image_raw/compressed"
        :param output_topic: 校正後影像 topic，例如 "/camera1_fix/color/image_raw/compressed"
        :param dim: 影像尺寸 (width, height)
        :param K: 相機內參數矩陣
        :param D: 畸變係數
        :param balance: 影像校正時平衡視角與黑邊，0 表示只保留無黑邊區域，1 表示保留全部視角
        """
        self.dim = dim
        self.K = K
        self.D = D
        self.bridge = CvBridge()

        # 計算新的相機矩陣，用於去畸變，調整 balance 參數可以保留較大的視角
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, self.dim, np.eye(3), balance=balance)
        
        # 訂閱原始影像
        self.sub = rospy.Subscriber(input_topic, CompressedImage, self.callback, queue_size=1)
        # 發佈校正後影像
        self.pub = rospy.Publisher(output_topic, CompressedImage, queue_size=1)
        
        rospy.loginfo("Initialized undistorter for topic: {}".format(input_topic))
        rospy.loginfo("New camera matrix:\n{}".format(self.new_K))

    def callback(self, msg):
        try:
            # Step 1: 轉換 CompressedImage 成 OpenCV 影像
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                rospy.logerr("Failed to decode image!")
                return

            # Step 2: 進行魚眼去畸變校正，使用 new_K 以保留較大的視角
            img_undistorted = cv2.fisheye.undistortImage(img, self.K, self.D, Knew=self.new_K)

            # Step 3: 轉換校正後影像回 ROS 訊息並發布
            msg_out = self.bridge.cv2_to_compressed_imgmsg(img_undistorted, dst_format="jpeg")
            self.pub.publish(msg_out)
        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))


if __name__ == '__main__':
    try:
        rospy.init_node('multi_fisheye_undistorter', anonymous=True)
        
        # 假設所有鏡頭解析度皆為 640x480，若有不同請分別設定
        DIM = (640, 480)

        # 鏡頭 1 的校正參數
        K1 = np.array([[263.80839843, 0.41589978, 327.54224715],
                       [0.0, 351.28264616, 238.17621743],
                       [0.0, 0.0, 1.0]])
        D1 = np.array([[-0.03004851], [0.08039913], [-0.14184036], [0.08669384]])

        # 鏡頭 2 的校正參數（例子，可根據實際數據調整）
        K2 = np.array([[263.80839843, 0.41589978, 327.54224715],
                       [0.0, 351.28264616, 238.17621743],
                       [0.0, 0.0, 1.0]])
        D2 = np.array([[-0.03004851], [0.08039913], [-0.14184036], [0.08669384]])

        # 鏡頭 3 的校正參數（例子，可根據實際數據調整）
        K3 = np.array([[263.80839843, 0.41589978, 327.54224715],
                       [0.0, 351.28264616, 238.17621743],
                       [0.0, 0.0, 1.0]])
        D3 = np.array([[-0.03004851], [0.08039913], [-0.14184036], [0.08669384]])

        # 分別建立三個 undistorter 實例，訂閱與發布各自的影像 topic
        # 將 balance 參數設為 1.0，保留全部視角（可能會有黑邊，可視需求調整）
        cam1 = CameraUndistorter(
            input_topic='/camera1/color/image_raw/compressed',
            output_topic='/camera1_fix/color/image_raw/compressed',
            dim=DIM, K=K1, D=D1, balance=0.47
        )
        cam2 = CameraUndistorter(
            input_topic='/camera2/color/image_raw/compressed',
            output_topic='/camera2_fix/color/image_raw/compressed',
            dim=DIM, K=K2, D=D2, balance=0.47
        )
        cam3 = CameraUndistorter(
            input_topic='/camera3/color/image_raw/compressed',
            output_topic='/camera3_fix/color/image_raw/compressed',
            dim=DIM, K=K3, D=D3, balance=0.47
        )

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()  # 確保退出時關閉所有 OpenCV 視窗
