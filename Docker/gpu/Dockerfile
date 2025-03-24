FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

ENV USER=leowang707
ENV HOME=/home/${USER}
ENV OPENCV_VERSION=4.5.2

# 更新系統時區與基礎套件
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 建立非root用戶
RUN useradd -m -s /bin/bash ${USER} && usermod -aG sudo ${USER} && \
    echo "root ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 安裝基本依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo build-essential cmake git unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    python3-dev python3-pip python3-numpy \
    python3-venv python3-setuptools \
    pciutils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 安裝 cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 安裝 GStreamer 相關套件（支援 OpenCV-GStreamer）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 檢測 GPU Compute Capability 並設置 CUDA_ARCH_BIN（僅將結果印出，不會直接用於 cmake 參數）
RUN export CUDA_ARCH_BIN=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '{printf "%.1f ", $1}') && \
    echo "Detected CUDA_ARCH_BIN: ${CUDA_ARCH_BIN}"

# 下載 OpenCV 與 OpenCV Contrib
RUN git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git /opencv
RUN git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git /opencv_contrib

# 建立 OpenCV 的 build 目錄
RUN mkdir -p /opencv/build
WORKDIR /opencv/build

# 編譯 OpenCV（包含 GStreamer 與 CUDA）
RUN export CUDA_ARCH_BIN=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '{printf "%.1f ", $1}') && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
          -D WITH_CUDA=ON \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUBLAS=1 \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D PYTHON_EXECUTABLE=/usr/bin/python3 \
          -D OPENCV_GENERATE_PKGCONFIG=YES \
          -D BUILD_opencv_python2=OFF \
          -D BUILD_opencv_python3=ON \
          -D WITH_GSTREAMER=ON \ 
          .. && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

# 測試 OpenCV 安裝是否成功 (含 CUDA)
RUN python3 -c "import cv2; print('OpenCV:', cv2.__version__, 'CUDA:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"

# 安裝 ROS Noetic
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release apt-transport-https ca-certificates

# 添加 ROS 軟體來源
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/ros-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -

# 安裝 ROS Noetic 套件
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-vision-opencv \
    python3-rosdep && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
    
# 初始化 rosdep
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
        rosdep init && \
        rosdep update --rosdistro noetic; \
    else \
        echo "rosdep 已經初始化過了"; \
    fi

# 安裝 ROS 其他常用套件
RUN pip3 install tqdm websocket-server websocket-client ffmpeg-python

# 安裝 ffplay 相關依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    libavdevice-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 安裝 rviz 及其相關依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-rviz \
    ros-noetic-rviz-visual-tools \
    ros-noetic-rqt-rviz \
    ros-noetic-image-view \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    ros-noetic-xacro \
    ros-noetic-urdf \
    ros-noetic-urdf-parser-plugin \
    ros-noetic-tf2-tools \
    ros-noetic-rqt \
    ros-noetic-rqt-common-plugins \
    ros-noetic-rqt-image-view \
    ros-noetic-rqt-robot-plugins \
    ros-noetic-rqt-tf-tree && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 安裝 image_transport 及影像插件
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-image-transport \
    ros-noetic-image-transport-plugins \
    ros-noetic-compressed-image-transport \
    ros-noetic-theora-image-transport && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 安裝 HTTP 上觀看 RTMP 串流之相關套件
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    vim \
    libnginx-mod-rtmp \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 安裝 RTSP 轉 HLS 串流之相關套件
ARG MTX_VERSION=1.11.3

RUN mkdir -p /opt/mediamtx && \
    curl -L https://github.com/bluenviron/mediamtx/releases/download/v${MTX_VERSION}/mediamtx_v${MTX_VERSION}_linux_amd64.tar.gz \
    | tar xz -C /opt/mediamtx && \
    ln -s /opt/mediamtx/mediamtx /usr/local/bin/mediamtx


# 設置 `bashrc` 以自動載入 ROS 環境與自定義路徑
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source ${HOME}/opencv-cuda-docker/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-docker/catkin_ws/src:\$ROS_PACKAGE_PATH" >> /root/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> /root/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> ${HOME}/.bashrc && \
    echo "source ${HOME}/opencv-cuda-docker/catkin_ws/devel/setup.bash" >> ${HOME}/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-docker/catkin_ws/src:\$ROS_PACKAGE_PATH" >> ${HOME}/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> ${HOME}/.bashrc && \
    echo "cp ${HOME}/opencv-cuda-docker/RTMP/nginx.conf /etc/nginx/nginx.conf" >> ~/.bashrc

# 再次設定用戶權限（確保 ${USER} 擁有 sudo 權限）
RUN echo "root ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

CMD ["bash"]
