# OpenCV CUDA Docker (and ROS)

OpenCV compiled with CUDA support for Docker.

## AMD64 Configuration

| Ubuntu  | CUDA  | OpenCV | ROS   |
|---------|-------|--------|-------|
| 20.04   | 11.3.1 | 4.5.2  | Noetic |

## ARM64 Configuration

| Ubuntu  | CUDA  | OpenCV | ROS   |
|---------|-------|--------|-------|
| 20.04   | 11.3.1 | 4.5.2  | Noetic |

---

## Usage Guide

### Start Image Stitching Processing
Execute the following command to start image stitching:

```bash
roslaunch image_processing image_stitcher_GPU.launch
```

### Start RTMP Streaming
Execute the following command to start RTMP streaming:

```bash
source RTMP/start_streaming.sh
```

### Optimize FPS Performance

#### In `image_stitcher_GPU`:
Modify the following parameters to enhance FPS:

```python
self.executor = ThreadPoolExecutor(max_workers=2)  # Increase for higher FPS, but monitor GPU performance.
interval = 0.1  # 10Hz - Decrease for higher FPS (used in rospy.Timer())
```

#### In `FLV_stream_Relay.py`:
Modify FFmpeg settings to improve FPS:

```bash
### ffmpeg setting ###
'-g', '6',  # Set GOP to 30 for better compression efficiency; increase for higher FPS.
