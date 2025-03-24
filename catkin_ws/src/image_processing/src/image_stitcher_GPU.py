#!/usr/bin/env python3
# File: image_stitcher_GPU.py

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from cv_bridge import CvBridge
import rospkg
import os
from Stitcher_GPU import Stitcher  # GPU 版
from collections import deque

# 引入 ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

rospack = rospkg.RosPack()

class ROSImageStitcher:
    def __init__(self):
        rospy.init_node('image_stitcher', anonymous=True)

        # 讀取 ROS 參數
        self.left_topic  = rospy.get_param('~sub_camera_topic_left',  '/camera3/color/image_raw/compressed')
        self.mid_topic   = rospy.get_param('~sub_camera_topic_mid',   '/camera2/color/image_raw/compressed')
        self.right_topic = rospy.get_param('~sub_camera_topic_right', '/camera1/color/image_raw/compressed')
        self.output_topic= rospy.get_param('~pub_camera_topic',       '/camera_stitched/color/image_raw/compressed')
        self.output_dir  = rospy.get_param('~output_dir',             'stitched_results')

        self.h1_path = rospy.get_param('~h1_path', "stitched_results/homography/640/H1_1.npy")
        self.h2_path = rospy.get_param('~h2_path', "stitched_results/homography/640/H2_1.npy")
        
        # 取得 Homography 檔案的完整路徑
        self.h1_path = os.path.join(rospack.get_path('image_processing'), self.h1_path) if self.h1_path else None
        self.h2_path = os.path.join(rospack.get_path('image_processing'), self.h2_path) if self.h2_path else None
        
        # 只讀取一次 H1 和 H2
        if self.h1_path and os.path.exists(self.h1_path):
            self.H1 = np.load(self.h1_path)
        else:
            self.H1 = None

        if self.h2_path and os.path.exists(self.h2_path):
            self.H2 = np.load(self.h2_path)
        else:
            self.H2 = None

        rospy.loginfo(f"Using homography files: {self.h1_path}, {self.h2_path}")

        # 建立影像轉換 Bridge
        self.bridge = CvBridge()

        # 初始化 image index
        self.image_index = 1

        # 建立三個佇列
        self.left_queue = deque(maxlen=6)
        self.mid_queue  = deque(maxlen=6)
        self.right_queue= deque(maxlen=6)

        # 建立 Publisher
        self.publisher = rospy.Publisher(self.output_topic, CompressedImage, queue_size=1)

        # 訂閱三個話題
        rospy.Subscriber(self.left_topic,  CompressedImage, self.left_callback)
        rospy.Subscriber(self.mid_topic,   CompressedImage, self.mid_callback)
        rospy.Subscriber(self.right_topic, CompressedImage, self.right_callback)

        # 初始化 GPU-based Stitcher
        self.stitcher = Stitcher()

        # 建立輸出資料夾
        self.full_output_dir = os.path.join(rospack.get_path('image_processing'), self.output_dir)
        os.makedirs(self.full_output_dir, exist_ok=True)
        self.intermediate_dir = os.path.join(self.full_output_dir, 'intermediate')
        self.homography_dir   = os.path.join(self.full_output_dir, 'homography')
        os.makedirs(self.intermediate_dir, exist_ok=True)
        os.makedirs(self.homography_dir, exist_ok=True)

        # Lock 用於保護佇列 (若需要更嚴謹可自行加強)
        self.lock = threading.Lock()

        # 建立 ThreadPoolExecutor，設定最多可用的 worker 數
        # 視實際需求可調整，例如 max_workers=2, 3, ...
        self.executor = ThreadPoolExecutor(max_workers=4)
        rospy.loginfo("ThreadPoolExecutor created with max_workers=2")

    def left_callback(self, msg):
        with self.lock:
            self.left_queue.append((msg.header.stamp, msg))

    def mid_callback(self, msg):
        with self.lock:
            self.mid_queue.append((msg.header.stamp, msg))

    def right_callback(self, msg):
        with self.lock:
            self.right_queue.append((msg.header.stamp, msg))

    def find_closest_match(self, timestamp, queue):
        """在給定的 queue 裡找離 timestamp 最近的影像。"""
        closest_msg = None
        min_diff = float('inf')
        for ts, msg in queue:
            diff = abs((timestamp - ts).to_sec())
            if diff < min_diff:
                closest_msg = msg
                min_diff = diff
        return closest_msg

    def stitch_images(self):
        """
        從 left_queue, mid_queue, right_queue 裡各取出能拼接的影像並進行拼接。
        回傳拼接後的影像 (numpy)，或 None 代表拼接失敗。
        """
        with self.lock:
            if not self.left_queue or not self.mid_queue or not self.right_queue:
                rospy.loginfo("Waiting for all images to arrive.")
                return None

            # 取出中相機最新影像作為 reference
            mid_timestamp, mid_msg = self.mid_queue.popleft()

        rospy.loginfo(f"[GPU Stitcher] Processing images for timestamp {mid_timestamp}.")

        # 在 lock 外（或內）尋找最接近的 left, right
        left_msg = None
        right_msg = None
        with self.lock:
            left_msg  = self.find_closest_match(mid_timestamp, self.left_queue)
            right_msg = self.find_closest_match(mid_timestamp, self.right_queue)

        if not left_msg or not right_msg:
            rospy.loginfo("Missing matching images for stitching.")
            return None

        # 轉成 OpenCV 圖片
        left_image  = self.bridge.compressed_imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
        mid_image   = self.bridge.compressed_imgmsg_to_cv2(mid_msg,   desired_encoding='bgr8')
        right_image = self.bridge.compressed_imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

        H1 = self.H1
        H2 = self.H2

        # CPU 翻轉
        img_left  = cv2.flip(mid_image, 1)
        img_right = cv2.flip(left_image, 1)

        # 1) 拼接 (Left & Mid)
        LM_img = self.stitcher.stitching(
            img_left=img_left,
            img_right=img_right,
            flip=True,
            H=H1,
            save_H_path=None if H1 is not None else os.path.join(self.homography_dir, f"H1_{self.image_index}.npy")
        )
        if LM_img is None:
            rospy.loginfo(f"Skipping stitching for image set {self.image_index} (H1 invalid).")
            return None

        # 若 H1/H2 都沒指定 => 動態學習 => 可存中間結果
        if not self.h1_path and not self.h2_path:
            intermediate_path = os.path.join(self.intermediate_dir, f"{self.image_index}.png")
            cv2.imwrite(intermediate_path, LM_img)

        # 2) 再拼接 (LM_img & right_image)
        final_image = self.stitcher.stitching(
            img_left=LM_img,
            img_right=right_image,
            flip=False,
            H=H2,
            save_H_path=None if H2 is not None else os.path.join(self.homography_dir, f"H2_{self.image_index}.npy")
        )
        if final_image is None:
            rospy.loginfo(f"Skipping final stitching for image set {self.image_index} (H2 invalid).")
            return None

        return final_image

    def process_images_task(self):
        """
        這個函式會被 ThreadPoolExecutor 的工作執行：
        - 從佇列取出影像，拼接，若成功則發布
        """
        stitched_image = self.stitch_images()
        if stitched_image is not None:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(stitched_image, dst_format='jpeg')
            self.publisher.publish(compressed_msg)
            rospy.loginfo(f"Published stitched image {self.image_index}.")
            self.image_index += 1

    def run(self):
        """
        使用 ThreadPoolExecutor 的方式:
        - 主線程進行 rospy.spin() 以維持回調
        - 同時定時提交 process_images_task() 到執行緒池執行
        """
        # 這裡我們用 rospy.Timer 或自己用 while 來定時提交任務
        # 1) 做一個 ROS Timer，每隔 0.1s (10Hz) 提交任務到 executor
        #   (或可用 threading.Timer、或在 while not rospy.is_shutdown() 中 sleep)
        hz = 30  # 想要 20Hz
        rospy.Timer(rospy.Duration(1.0 / hz), self.timer_callback)
        rospy.loginfo("Main thread: start spin() for callbacks.")
        rospy.spin()
        rospy.loginfo("Main thread: spin() ended, shutting down executor.")
        self.executor.shutdown(wait=True)

    def timer_callback(self, event):
        """
        以 ROS Timer 的方式，週期性呼叫。每次呼叫就提交一次拼接任務到執行緒池。
        """
        # 提交工作，讓執行緒池背景執行
        self.executor.submit(self.process_images_task)


if __name__ == '__main__':
    try:
        stitcher = ROSImageStitcher()
        stitcher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("GPU-based Image stitcher node terminated.")

# origin version (without ThreadPoolExecutor)
# #!/usr/bin/env python3
# # File: image_stitcher_GPU.py

# import rospy
# from sensor_msgs.msg import CompressedImage
# import cv2
# import numpy as np
# from cv_bridge import CvBridge
# import rospkg
# import os
# from Stitcher_GPU import Stitcher  # GPU 版
# from collections import deque

# rospack = rospkg.RosPack()

# class ROSImageStitcher:
#     def __init__(self):
#         # Initialize ROS node
#         rospy.init_node('image_stitcher', anonymous=True)

#         # Load parameters
#         self.left_topic = rospy.get_param('~sub_camera_topic_left', '/camera3/color/image_raw/compressed')
#         self.mid_topic  = rospy.get_param('~sub_camera_topic_mid',  '/camera2/color/image_raw/compressed')
#         self.right_topic= rospy.get_param('~sub_camera_topic_right','/camera1/color/image_raw/compressed')
#         self.output_topic = rospy.get_param('~pub_camera_topic', '/camera_stitched/color/image_raw/compressed')
#         self.output_dir   = rospy.get_param('~output_dir', 'stitched_results')

#         self.h1_path = rospy.get_param('~h1_path', "stitched_results/homography/640/H1_1.npy")
#         self.h2_path = rospy.get_param('~h2_path', "stitched_results/homography/640/H2_1.npy")
        
#         # 取得 Homography 檔案的完整路徑
#         self.h1_path = os.path.join(rospack.get_path('image_processing'), self.h1_path) if self.h1_path else None
#         self.h2_path = os.path.join(rospack.get_path('image_processing'), self.h2_path) if self.h2_path else None

#         # 只讀取一次 H1 和 H2
#         if self.h1_path and os.path.exists(self.h1_path):
#             self.H1 = np.load(self.h1_path)
#         else:
#             self.H1 = None

#         if self.h2_path and os.path.exists(self.h2_path):
#             self.H2 = np.load(self.h2_path)
#         else:
#             self.H2 = None

#         rospy.loginfo(f"Using homography files: {self.h1_path}, {self.h2_path}")

#         # Bridge for converting ROS images to OpenCV
#         self.bridge = CvBridge()

#         # Initialize the image index
#         self.image_index = 1

#         # Queues to store incoming images
#         self.left_queue = deque(maxlen=6)
#         self.mid_queue  = deque(maxlen=6)
#         self.right_queue= deque(maxlen=6)

#         # Publisher
#         self.publisher = rospy.Publisher(self.output_topic, CompressedImage, queue_size=1)

#         # Subscribers
#         rospy.Subscriber(self.left_topic,  CompressedImage, self.left_callback)
#         rospy.Subscriber(self.mid_topic,   CompressedImage, self.mid_callback)
#         rospy.Subscriber(self.right_topic, CompressedImage, self.right_callback)

#         rospy.loginfo("GPU-based Image stitcher initialized. Waiting for images...")

#     def left_callback(self, msg):
#         self.left_queue.append((msg.header.stamp, msg))

#     def mid_callback(self, msg):
#         self.mid_queue.append((msg.header.stamp, msg))

#     def right_callback(self, msg):
#         self.right_queue.append((msg.header.stamp, msg))

#     def find_closest_match(self, timestamp, queue):
#         """Find the closest image in the queue based on timestamp."""
#         closest_msg = None
#         min_diff = float('inf')
#         for ts, msg in queue:
#             diff = abs((timestamp - ts).to_sec())
#             if diff < min_diff:
#                 closest_msg = msg
#                 min_diff = diff
#         return closest_msg

#     def stitch_images(self):
#         if not self.left_queue or not self.mid_queue or not self.right_queue:
#             rospy.loginfo("Waiting for all images to arrive.")
#             return None

#         # Use the middle camera's latest image as reference
#         mid_timestamp, mid_msg = self.mid_queue.popleft()
#         rospy.loginfo(f"[GPU Stitcher] Processing images for timestamp {mid_timestamp}.")

#         # Find the closest images in left and right queues
#         left_msg = self.find_closest_match(mid_timestamp, self.left_queue)
#         right_msg = self.find_closest_match(mid_timestamp, self.right_queue)

#         if not left_msg or not right_msg:
#             rospy.loginfo("Missing matching images for stitching.")
#             return None

#         # Convert ROS image messages to OpenCV images (CPU)
#         left_image = self.bridge.compressed_imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
#         mid_image  = self.bridge.compressed_imgmsg_to_cv2(mid_msg,  desired_encoding='bgr8')
#         right_image= self.bridge.compressed_imgmsg_to_cv2(right_msg,desired_encoding='bgr8')

#         # 建立輸出目錄
#         output_dir = os.path.join(rospack.get_path('image_processing'), self.output_dir)
#         os.makedirs(output_dir, exist_ok=True)

#         intermediate_dir = os.path.join(output_dir, 'intermediate')
#         homography_dir   = os.path.join(output_dir, 'homography')
#         os.makedirs(intermediate_dir, exist_ok=True)
#         os.makedirs(homography_dir, exist_ok=True)

#         # Initialize the GPU-based stitcher
#         stitcher = Stitcher()
        
#         # 直接沿用 self.H1, self.H2，不重複讀取
#         H1 = self.H1
#         H2 = self.H2

#         # 這裡仍在 CPU 做翻轉，如要 GPU flip，請自行改寫
#         img_left  = cv2.flip(mid_image, 1)
#         img_right = cv2.flip(left_image, 1)

#         # =============== 第一階段拼接 (Left & Mid) ===============
#         LM_img = stitcher.stitching(
#             img_left=img_left,
#             img_right=img_right,
#             flip=True,
#             H=H1,
#             # 若 H1=None => 動態學 => 存檔
#             save_H_path=None if H1 is not None else os.path.join(homography_dir, f"H1_{self.image_index}.npy")
#         )
#         if LM_img is None:
#             rospy.loginfo(f"Skipping stitching for image set {self.image_index} (H1 invalid).")
#             return None

#         # 若 H1/H2 都沒指定 => 代表要動態學習 => 可存中間結果
#         if not self.h1_path and not self.h2_path:
#             intermediate_path = os.path.join(intermediate_dir, f"{self.image_index}.png")
#             cv2.imwrite(intermediate_path, LM_img)

#         # =============== 第二階段拼接 (LM_img & Right) ===============
#         final_image = stitcher.stitching(
#             img_left=LM_img,
#             img_right=right_image,
#             flip=False,
#             H=H2,
#             save_H_path=None if H2 is not None else os.path.join(homography_dir, f"H2_{self.image_index}.npy")
#         )
#         if final_image is None:
#             rospy.loginfo(f"Skipping final stitching for image set {self.image_index} (H2 invalid).")
#             return None

#         return final_image

#     def process_images(self):
#         stitched_image = self.stitch_images()
#         if stitched_image is not None:
#             compressed_msg = self.bridge.cv2_to_compressed_imgmsg(stitched_image, dst_format='jpeg')
#             self.publisher.publish(compressed_msg)
#             rospy.loginfo(f"Published stitched image {self.image_index}.")
#             self.image_index += 1  # 更新計數

#     def run(self):
#         rate = rospy.Rate(10)  # 10 Hz
#         while not rospy.is_shutdown():
#             self.process_images()
#             rate.sleep()

# if __name__ == '__main__':
#     try:
#         stitcher = ROSImageStitcher()
#         stitcher.run()
#     except rospy.ROSInterruptException:
#         rospy.loginfo("GPU-based Image stitcher node terminated.")
