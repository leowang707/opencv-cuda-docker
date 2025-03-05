#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from cv_bridge import CvBridge
import rospkg
import os
from Stitcher_GPU import Stitcher  # 使用 GPU 加速的 Stitcher
from collections import deque

rospack = rospkg.RosPack()

class ROSImageStitcher:
    def __init__(self):
        rospy.init_node('image_stitcher', anonymous=True)

        self.left_topic = rospy.get_param('~sub_camera_topic_left', '/camera3/color/image_raw/compressed')
        self.mid_topic = rospy.get_param('~sub_camera_topic_mid', '/camera2/color/image_raw/compressed')
        self.right_topic = rospy.get_param('~sub_camera_topic_right', '/camera1/color/image_raw/compressed')
        self.output_topic = rospy.get_param('~pub_camera_topic', '/camera_stitched/color/image_raw/compressed')
        self.output_dir = rospy.get_param('~output_dir', 'stitched_results')
        self.h1_path = rospy.get_param('~h1_path', "stitched_results/homography/640/H1_1.npy")
        self.h2_path = rospy.get_param('~h2_path', "stitched_results/homography/640/H2_1.npy")

        self.bridge = CvBridge()
        self.image_index = 1
        self.left_queue = deque(maxlen=1)
        self.mid_queue = deque(maxlen=1)
        self.right_queue = deque(maxlen=1)
        self.publisher = rospy.Publisher(self.output_topic, CompressedImage, queue_size=1)

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

        mid_timestamp, mid_msg = self.mid_queue.popleft()
        left_msg = self.find_closest_match(mid_timestamp, self.left_queue)
        right_msg = self.find_closest_match(mid_timestamp, self.right_queue)

        if not left_msg or not right_msg:
            rospy.loginfo("Missing matching images for stitching.")
            return None

        left_image = self.bridge.compressed_imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        mid_image = self.bridge.compressed_imgmsg_to_cv2(mid_msg, desired_encoding='bgr8')
        right_image = self.bridge.compressed_imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

        stitcher = Stitcher()

        gpu_left = cv2.cuda_GpuMat()
        gpu_mid = cv2.cuda_GpuMat()
        gpu_right = cv2.cuda_GpuMat()

        gpu_left.upload(left_image)
        gpu_mid.upload(mid_image)
        gpu_right.upload(right_image)

        img_left = cv2.cuda.flip(gpu_mid, 1).download()
        img_right = cv2.cuda.flip(gpu_left, 1).download()

        H1 = np.load(self.h1_path) if os.path.exists(self.h1_path) else None
        H2 = np.load(self.h2_path) if os.path.exists(self.h2_path) else None

        LM_img = stitcher.stitching(img_left, img_right, flip=True, H=H1)
        if LM_img is None:
            rospy.loginfo("Skipping stitching due to missing H1.")
            return None

        final_image = stitcher.stitching(LM_img, right_image, flip=False, H=H2)
        if final_image is None:
            rospy.loginfo("Skipping final stitching due to missing H2.")
            return None

        return final_image

    def process_images(self):
        stitched_image = self.stitch_images()
        if stitched_image is not None:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(stitched_image, dst_format='jpeg')
            self.publisher.publish(compressed_msg)
            rospy.loginfo(f"Published stitched image {self.image_index}.")
            self.image_index += 1

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.process_images()
            rate.sleep()

if __name__ == '__main__':
    try:
        stitcher = ROSImageStitcher()
        stitcher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("GPU-based Image stitcher node terminated.")
