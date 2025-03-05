#!/usr/bin/env python3

import cv2
import numpy as np

def check_opencv_cuda():
    print("\nChecking OpenCV CUDA Support...")
    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("No CUDA-capable device found. Please check your installation.")
        return

    print("\nRunning GPU Processing Test...")
    
    # 創建隨機影像
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    print("Original Image Shape:", img.shape)
    
    # 嘗試上傳到 GPU
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    print("Image uploaded to GPU successfully.")
    
    # 改用 `createGaussianFilter`
    gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 0)
    gpu_blurred = gaussian_filter.apply(gpu_img)
    print("Gaussian blur applied on GPU successfully.")

    # 下載回 CPU
    img_blurred = gpu_blurred.download()
    print("Image downloaded from GPU successfully. Shape:", img_blurred.shape)
    
    print("\nCUDA Test Completed Successfully.")

if __name__ == "__main__":
    check_opencv_cuda()
