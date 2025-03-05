#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from cv_bridge import CvBridge
import rospkg
import os
# 原本是 from Stitcher import Stitcher，改為用 GPU 版
from Stitcher_GPU import Stitcher
from collections import deque

rospack = rospkg.RosPack()

class ROSImageStitcher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('image_stitcher', anonymous=True)

        # Load parameters
        self.left_topic = rospy.get_param('~sub_camera_topic_left', '/camera3/color/image_raw/compressed')
        self.mid_topic = rospy.get_param('~sub_camera_topic_mid', '/camera2/color/image_raw/compressed')
        self.right_topic = rospy.get_param('~sub_camera_topic_right', '/camera1/color/image_raw/compressed')
        self.output_topic = rospy.get_param('~pub_camera_topic', '/camera_stitched/color/image_raw/compressed')
        self.output_dir = rospy.get_param('~output_dir', 'stitched_results')
        self.h1_path = rospy.get_param('~h1_path', "stitched_results/homography/640/H1_1.npy")
        self.h2_path = rospy.get_param('~h2_path', "stitched_results/homography/640/H2_1.npy")

        # Bridge for converting ROS images to OpenCV
        self.bridge = CvBridge()

        # Initialize the image index
        self.image_index = 1

        # Queues to store incoming images
        self.left_queue = deque(maxlen=1)
        self.mid_queue = deque(maxlen=1)
        self.right_queue = deque(maxlen=1)

        # Publisher
        self.publisher = rospy.Publisher(self.output_topic, CompressedImage, queue_size=1)

        # Subscribers
        rospy.Subscriber(self.left_topic, CompressedImage, self.left_callback)
        rospy.Subscriber(self.mid_topic, CompressedImage, self.mid_callback)
        rospy.Subscriber(self.right_topic, CompressedImage, self.right_callback)

        rospy.loginfo("GPU-based Image stitcher initialized. Waiting for images...")

    def left_callback(self, msg):
        self.left_queue.append((msg.header.stamp, msg))

    def mid_callback(self, msg):
        self.mid_queue.append((msg.header.stamp, msg))

    def right_callback(self, msg):
        self.right_queue.append((msg.header.stamp, msg))

    def find_closest_match(self, timestamp, queue):
        """Find the closest image in the queue based on timestamp."""
        closest_msg = None
        min_diff = float('inf')
        for ts, msg in queue:
            diff = abs((timestamp - ts).to_sec())
            if diff < min_diff:
                closest_msg = msg
                min_diff = diff
        return closest_msg

    def stitch_images(self):
        if not self.left_queue or not self.mid_queue or not self.right_queue:
            rospy.loginfo("Waiting for all images to arrive.")
            return None

        # Use the middle camera's latest image as the reference
        mid_timestamp, mid_msg = self.mid_queue.popleft()
        rospy.loginfo(f"[GPU Stitcher] Processing images for timestamp {mid_timestamp}.")

        # Find the closest images in left and right queues
        left_msg = self.find_closest_match(mid_timestamp, self.left_queue)
        right_msg = self.find_closest_match(mid_timestamp, self.right_queue)

        if not left_msg or not right_msg:
            rospy.loginfo("Missing matching images for stitching.")
            return None

        # Convert ROS image messages to OpenCV images (CPU)
        left_image = self.bridge.compressed_imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        mid_image = self.bridge.compressed_imgmsg_to_cv2(mid_msg, desired_encoding='bgr8')
        right_image = self.bridge.compressed_imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

        # 建立輸出目錄
        output_dir = os.path.join(rospack.get_path('image_processing'), self.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        intermediate_dir = os.path.join(output_dir, 'intermediate')
        homography_dir = os.path.join(output_dir, 'homography')
        os.makedirs(intermediate_dir, exist_ok=True)
        os.makedirs(homography_dir, exist_ok=True)

        # Initialize the GPU-based stitcher
        stitcher = Stitcher()

        # 找到 Homography 檔路徑，若無則為 None
        h1_path = os.path.join(rospack.get_path('image_processing'), self.h1_path) if self.h1_path else None
        h2_path = os.path.join(rospack.get_path('image_processing'), self.h2_path) if self.h2_path else None
        rospy.loginfo(f"Using homography files: {h1_path}, {h2_path}")
        H1 = np.load(h1_path) if h1_path and os.path.exists(h1_path) else None
        H2 = np.load(h2_path) if h2_path and os.path.exists(h2_path) else None

        # 這裡仍在 CPU 上做翻轉，如要 GPU flip，請自行改成 cv2.cuda.flip
        img_left = cv2.flip(mid_image, 1)
        img_right = cv2.flip(left_image, 1)

        # =============== 第一階段拼接 (Left & Mid) ===============
        LM_img = stitcher.stitching(
            img_left,
            img_right,
            flip=True,
            H=H1,
            save_H_path=(None if H1 is not None else os.path.join(homography_dir, f"H1_{self.image_index}.npy"))
        )

        if LM_img is None:
            rospy.loginfo(f"Skipping stitching for image set {self.image_index} due to missing/invalid stitching.")
            return None

        # 若 H1/H2 都沒指定，代表要動態學，存個中間結果
        if not self.h1_path and not self.h2_path:
            intermediate_path = os.path.join(intermediate_dir, f"{self.image_index}.png")
            cv2.imwrite(intermediate_path, LM_img)

        # 接著要把剛拼好的 LM_img (左+中) 與 right_image 拼接
        img_left = LM_img
        img_right = right_image

        # =============== 第二階段拼接 (上一步結果 & Right) ===============
        final_image = stitcher.stitching(
            img_left,
            img_right,
            flip=False,
            H=H2,
            save_H_path=(None if H2 is not None else os.path.join(homography_dir, f"H2_{self.image_index}.npy"))
        )

        if final_image is None:
            rospy.loginfo(f"Skipping final stitching for image set {self.image_index} due to missing/invalid stitching.")
            return None

        return final_image

    def process_images(self):
        stitched_image = self.stitch_images()
        if stitched_image is not None:
            # Convert OpenCV image to ROS CompressedImage message
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(stitched_image, dst_format='jpeg')
            self.publisher.publish(compressed_msg)
            rospy.loginfo(f"Published stitched image {self.image_index - 1}.")
            self.image_index += 1  # 記得更新計數

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.process_images()
            rate.sleep()


if __name__ == '__main__':
    try:
        stitcher = ROSImageStitcher()
        stitcher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("GPU-based Image stitcher node terminated.")
