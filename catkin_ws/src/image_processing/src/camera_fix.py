#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class CameraUndistorter:
    def __init__(self, input_topic, output_topic, dim, K, D, balance=1.0):
        self.dim = dim
        self.K = K
        self.D = D
        self.bridge = CvBridge()

        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, self.dim, np.eye(3), balance=balance)

        self.sub = rospy.Subscriber(input_topic, CompressedImage, self.callback, queue_size=1)
        self.pub = rospy.Publisher(output_topic, CompressedImage, queue_size=1)

        rospy.loginfo(f"[{input_topic}] Undistorter initialized.")
        rospy.loginfo(f"[{input_topic}] New camera matrix:\n{self.new_K}")

    def callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                rospy.logerr("Failed to decode image.")
                return

            img_undistorted = cv2.fisheye.undistortImage(img, self.K, self.D, Knew=self.new_K)
            msg_out = self.bridge.cv2_to_compressed_imgmsg(img_undistorted, dst_format="jpeg")
            self.pub.publish(msg_out)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        rospy.init_node('multi_fisheye_undistorter', anonymous=True)

        # 共用的相機參數
        DIM = (640, 480)
        K = np.array([[263.80839843, 0.41589978, 327.54224715],
                      [0.0, 351.28264616, 238.17621743],
                      [0.0, 0.0, 1.0]])
        D = np.array([[-0.03004851], [0.08039913], [-0.14184036], [0.08669384]])

        # 相機 topic 配對清單（原始 topic, 校正後 topic）
        camera_pairs = [
            ('/camera1/color/image_raw/compressed', '/camera1_fix/color/image_raw/compressed'),
            ('/camera2/color/image_raw/compressed', '/camera2_fix/color/image_raw/compressed'),
            ('/camera3/color/image_raw/compressed', '/camera3_fix/color/image_raw/compressed'),
            ('/camera4/color/image_raw/compressed', '/camera4_fix/color/image_raw/compressed'),
        ]

        # 建立多個 undistorter
        undistorters = [
            CameraUndistorter(in_topic, out_topic, DIM, K, D, balance=0.47)
            for in_topic, out_topic in camera_pairs
        ]

        rospy.loginfo("✅ 四台相機魚眼去畸變模組已啟動")
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
