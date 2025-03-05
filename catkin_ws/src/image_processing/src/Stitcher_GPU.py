#!/usr/bin/env python3

import os
import cv2
import numpy as np
import argparse
import time
from tqdm import tqdm

class Setting():
    def __init__(self, left_dir, mid_dir, right_dir, output_dir):
        self.left_dir = left_dir
        self.mid_dir = mid_dir
        self.right_dir = right_dir
        self.output_dir = output_dir
    
    def check_and_make_dir(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
    
    def load_images(self, dir_path):
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist.")
            return []
        
        images = []
        files = sorted([f for f in os.listdir(dir_path) if f.endswith('.png')])
        if not files:
            print(f"Warning: No valid images found in {dir_path}.")
            return []
        
        for file in files:
            img_path = os.path.join(dir_path, file)
            image = cv2.imread(img_path)
            if image is not None:
                images.append(image)
            else:
                print(f"Warning: Failed to load image {img_path}.")
        return images

    
    def file_setting(self):
        left_images = self.load_images(self.left_dir)
        base_images = self.load_images(self.mid_dir)
        right_images = self.load_images(self.right_dir)
        
        output_images_dir = os.path.join(self.output_dir, 'images')
        self.check_and_make_dir(output_images_dir)
        
        return left_images, base_images, right_images, output_images_dir

class Stitcher():
    def __init__(self):
        pass

    def remove_black_border(self, img):
        """
        移除影像周圍的黑邊，以尋找最大有效區域。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_rect = (0, 0, img.shape[1], img.shape[0])
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > max_area:
                max_area = area
                best_rect = (x, y, w, h)
        x, y, w, h = best_rect
        return img[y:y+h, x:x+w]

    def linearBlending(self, img_left, img_right, use_blending=True):
        """
        使用 GPU 加速的 Alpha Blending (方法二)：
        1. 在重疊區域使用 alpha = 1->0 的線性權重
        2. 非重疊區域保持原樣，避免亮度被壓低

        :param img_left: 左影像 (H1 x W1 x 3)
        :param img_right: 右影像 (H2 x W2 x 3)
        :param use_blending: 是否啟用 Blending
        :return: 融合後的影像 (uint8)
        """
        if not use_blending:
            # 不使用 blending，直接右圖疊在左圖上
            result = img_left.copy()
            result[:img_right.shape[0], :img_right.shape[1]] = img_right
            return result

        # --- 1) 建立大畫布，容納左右圖 ---
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]
        height = max(h1, h2)
        width = max(w1, w2)

        img_left_large = np.zeros((height, width, 3), dtype=np.uint8)
        img_right_large = np.zeros((height, width, 3), dtype=np.uint8)

        img_left_large[:h1, :w1]  = img_left
        img_right_large[:h2, :w2] = img_right

        # --- 2) 找出重疊區域 ---
        overlap_mask = np.logical_and(
            np.any(img_left_large  != 0, axis=2),
            np.any(img_right_large != 0, axis=2)
        )
        overlap_indices = np.where(np.any(overlap_mask, axis=0))[0]
        if len(overlap_indices) < 2:
            # 沒有重疊，直接合併並回傳
            return img_left_large + img_right_large

        min_x, max_x = overlap_indices[0], overlap_indices[-1]
        overlap_width = max_x - min_x + 1

        # --- 3) 建立 alpha_mask (單通道) ---
        alpha_mask = np.zeros((height, width), dtype=np.float32)
        alpha_line = np.linspace(1.0, 0.0, overlap_width).astype(np.float32)
        alpha_mask[:, min_x:max_x+1] = np.tile(alpha_line, (height, 1))

        # --- 4) 將影像與 alpha 上傳到 GPU ---
        # 先轉成 float32，才能正確執行乘法
        gpu_left = cv2.cuda_GpuMat()
        gpu_right = cv2.cuda_GpuMat()
        gpu_left.upload(img_left_large.astype(np.float32))
        gpu_right.upload(img_right_large.astype(np.float32))

        # *** 關鍵：在 CPU 合成 3 通道 alpha，再上傳 GPU ***
        alpha_3c = cv2.merge([alpha_mask, alpha_mask, alpha_mask])  # shape: (H, W, 3)
        gpu_alpha_3c = cv2.cuda_GpuMat()
        gpu_alpha_3c.upload(alpha_3c)  # 這樣就可以保證是 GPU GpuMat

        # --- 5) 計算 left * alpha + right * (1 - alpha) ---
        # out_left = gpu_left * alpha
        out_left = cv2.cuda.multiply(gpu_left, gpu_alpha_3c)

        # 建一張全為 1 的圖 (H, W, 3)，用於 (1 - alpha)
        gpu_one = cv2.cuda_GpuMat(gpu_alpha_3c.size(), gpu_alpha_3c.type())
        one_np = np.ones((height, width, 3), dtype=np.float32)
        gpu_one.upload(one_np)

        inv_alpha_3c = cv2.cuda.subtract(gpu_one, gpu_alpha_3c)  # (1 - alpha)
        out_right = cv2.cuda.multiply(gpu_right, inv_alpha_3c)

        blended_gpu = cv2.cuda.add(out_left, out_right)
        blended = blended_gpu.download().astype(np.uint8)

        # --- 6) 修正非重疊區域：直接保留原圖 ---
        # overlap_mask=False => 那裡只有單方像素 (或全黑)
        blended[~overlap_mask] = (
            img_left_large[~overlap_mask] + img_right_large[~overlap_mask]
        )

        return blended

    def stitching(self, img_left, img_right, flip=False, H=None, save_H_path=None, use_blending=True):
        """
        影像拼接流程：
        1. 若無 H，透過 SIFT/BF/RANSAC 求得 H
        2. GPU warpPerspective 將右圖投影到左圖
        3. 使用線性AlphaBlending進行影像融合
        4. 移除黑邊，翻轉 (可選)
        """
        # === 1) 若沒有提供 H，自動計算 ===
        if H is None:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img_left, None)
            kp2, des2 = sift.detectAndCompute(img_right, None)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            good_matches = []
            src_pts = []
            dst_pts = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    src_pts.append(kp1[m.queryIdx].pt)
                    dst_pts.append(kp2[m.trainIdx].pt)

            src_pts = np.float32(src_pts)
            dst_pts = np.float32(dst_pts)

            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if save_H_path is not None and H is not None:
                np.save(save_H_path, H)

        # === 2) 在 GPU 上對右圖做 warpPerspective ===
        gpu_right = cv2.cuda_GpuMat()
        gpu_right.upload(img_right)
        warped_size = (img_left.shape[1] + img_right.shape[1], img_left.shape[0])
        gpu_right_warped = cv2.cuda.warpPerspective(gpu_right, H, warped_size)
        img_right_warped = gpu_right_warped.download()

        # === 3) 把左圖與變形後的右圖合到同一張大圖上，再用 alpha blending ===
        panorama = np.zeros_like(img_right_warped)
        panorama[:img_left.shape[0], :img_left.shape[1]] = img_left

        blended = self.linearBlending(panorama, img_right_warped, use_blending)

        # === 4) 去黑邊 & flip (可選) ===
        cropped_result = self.remove_black_border(blended)
        if flip:
            cropped_result = cv2.flip(cropped_result, 1)
        return cropped_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', type=str, required=True)
    parser.add_argument('--mid', type=str, required=True)
    parser.add_argument('--right', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--h1_path', type=str, default=None)
    parser.add_argument('--h2_path', type=str, default=None)
    parser.add_argument('--no_blending', action='store_true', help="Disable blending for stitching")

    args = parser.parse_args()

    # 建立 Setting，讀取影像
    setting = Setting(args.left, args.mid, args.right, args.output_dir)
    left_images, mid_images, right_images, output_images_dir = setting.file_setting()

    # 建立 Stitcher
    stitcher = Stitcher()

    # 判斷是否要關閉 Blending
    use_blending = not args.no_blending

    # 逐張影像執行拼接
    for i in range(len(left_images)):
        # 讀取 H1, H2
        H1 = np.load(args.h1_path) if args.h1_path and os.path.exists(args.h1_path) else None
        H2 = np.load(args.h2_path) if args.h2_path and os.path.exists(args.h2_path) else None

        # 1) 中圖 & 左圖拼接 (翻轉左圖)
        LM_img = stitcher.stitching(
            img_left=mid_images[i],
            img_right=left_images[i],
            flip=True,
            H=H1,
            use_blending=use_blending
        )

        # 2) 上一步結果再跟右圖拼接
        final_image = stitcher.stitching(
            img_left=LM_img,
            img_right=right_images[i],
            flip=False,
            H=H2,
            use_blending=use_blending
        )

        # 輸出結果
        final_path = os.path.join(output_images_dir, f"{i+1}.png")
        cv2.imwrite(final_path, final_image)
        print(f"[INFO] Saved: {final_path}")
