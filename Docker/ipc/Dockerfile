# 使用 NVIDIA Jetson L4T 基礎映像 (適用於 Jetson Orin)
FROM nvcr.io/nvidia/l4t-base:r35.3.1

# 設定 Shell 行為
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# 定義用戶與家目錄
ENV USER=jetson
ENV HOME=/home/${USER}
ENV OPENCV_VERSION=4.5.2

# 更新系統並安裝必要套件
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo wget curl git cmake unzip pkg-config \
    build-essential python3-pip python3-numpy python3-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    ffmpeg libavutil-dev libavfilter-dev \
    libavdevice-dev vim nano && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 安裝 cuDNN (Jetson 已內建 CUDA)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 建立非 root 用戶
RUN useradd -m -s /bin/bash ${USER} && usermod -aG sudo ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 下載 OpenCV 及 Contrib
RUN git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git /opencv
RUN git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git /opencv_contrib

# 建立 OpenCV 構建目錄
RUN mkdir -p /opencv/build
WORKDIR /opencv/build

# 編譯 OpenCV
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
          -D WITH_CUDA=ON \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUBLAS=1 \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D PYTHON_EXECUTABLE=/usr/bin/python3 \
          -D OPENCV_GENERATE_PKGCONFIG=YES \
          -D BUILD_opencv_python2=OFF \
          -D BUILD_opencv_python3=ON \
          .. && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

# 測試 OpenCV 是否正確編譯
RUN python3 -c "import cv2; print('OpenCV:', cv2.__version__, 'CUDA:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"

# 安裝 ROS Noetic
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release apt-transport-https ca-certificates && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/ros-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-vision-opencv \
    python3-rosdep && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 初始化 rosdep
RUN rosdep init && rosdep update --rosdistro noetic

# 安裝 ROS 依賴
RUN pip3 install tqdm websocket-server websocket-client ffmpeg-python

# 安裝 RTMP 串流相關套件
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx libnginx-mod-rtmp && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 複製 NGINX 設定檔
COPY nginx.conf /etc/nginx/nginx.conf

# 設定 `bashrc` 以自動載入 ROS 環境
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> ${HOME}/.bashrc && \
    echo "cp ${HOME}/opencv-cuda-docker/RTMP/nginx.conf /etc/nginx/nginx.conf" >> ~/.bashrc

# 設置用戶權限
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 設置啟動指令
CMD ["bash"]
