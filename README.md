# OpenCV CUDA Docker (and ROS)

OpenCV compiled with CUDA support for Docker.

## AMD64

|Ubuntu|CUDA|OpenCV|ROS|
|---|---|---|---|
|20.04|11.3.1|4.5.2|Noetic|

## ARM64

|Ubuntu|CUDA|OpenCV|ROS|
|---|---|---|---|
|20.04|11.3.1|4.5.2|Noetic|

## 使用方式

### 啟動影像拼接處理
執行以下指令來啟動影像拼接：
```bash
roslaunch image_processing image_stitcher_GPU.launch
```

### 啟動 RTMP 串流
執行以下指令來啟動 RTMP 串流：
```bash
source RTMP/start_streaming.sh
```