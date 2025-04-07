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
        
        # 每台相機的參數（K, D）
        camera_configs = [
            # camera left
            (
                '/camera1/color/image_raw/compressed',
                '/camera1_fix/color/image_raw/compressed',
                np.array([[2.94104211e+02, 1.00501200e-01, 3.17351129e+02],
                          [0.00000000e+00, 3.90626759e+02, 2.44645859e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                np.array([[-0.15439177], [ 0.45612835], [-0.79521684], [ 0.46727377]])
            ),
            # camera middle
            (
                '/camera2/color/image_raw/compressed',
                '/camera2_fix/color/image_raw/compressed',
                np.array([[ 2.62450733e+02, -1.42390413e-01,  3.24028455e+02],
                          [ 0.00000000e+00,  3.48556538e+02,  2.56079733e+02],
                          [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
                np.array([[-0.01878612], [ 0.02131176], [-0.02093047], [ 0.00497547]])
            ),
            # camera right
            (
                '/camera3/color/image_raw/compressed',
                '/camera3_fix/color/image_raw/compressed',
                np.array([[ 2.60945033e+02, -2.08546212e-01,  3.24930850e+02],
                          [ 0.00000000e+00,  3.47072705e+02,  2.62938882e+02],
                          [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
                np.array([[-0.03237013], [ 0.0271175 ], [-0.01698575], [ 0.00307905]])
            ),
            # camera back
            # (
            #     '/camera4/color/image_raw/compressed',
            #     '/camera4_fix/color/image_raw/compressed',
            #     np.array([[295.0, 0.1, 318.0],
            #               [0.0, 395.0, 245.0],
            #               [0.0, 0.0, 1.0]]),
            #     np.array([[-0.158], [0.458], [-0.798], [0.465]])
            # )
        ]

        undistorters = [
            CameraUndistorter(in_topic, out_topic, DIM, K, D, balance=0.41)
            for in_topic, out_topic, K, D in camera_configs
        ]

        rospy.loginfo("✅ 四台相機魚眼去畸變模組已啟動")
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
