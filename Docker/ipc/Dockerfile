# 使用 NVIDIA Jetson L4T (r35.3.1) 基底映像
FROM nvcr.io/nvidia/l4t-base:r35.3.1

# ------------------------------------------------------------------------------
# Dockerfile 說明
# ------------------------------------------------------------------------------
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

ENV USER=leowang707
ENV HOME=/home/${USER}
ENV OPENCV_VERSION=4.5.2
# Jetson Orin Ampere GPU Compute Capability (8.7)
ENV CUDA_ARCH_BIN=8.7

# ------------------------------------------------------------------------------
# 1. 系統更新，安裝必要套件
# ------------------------------------------------------------------------------
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 建立非 root 用戶
RUN useradd -m -s /bin/bash ${USER} && usermod -aG sudo ${USER} && \
    echo "root ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ------------------------------------------------------------------------------
# 2. 安裝編譯 OpenCV 所需的基本依賴
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    python3-dev python3-pip python3-numpy \
    python3-venv python3-setuptools \
    pciutils curl gnupg2 lsb-release apt-transport-https \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Jetson 上通常已內建 CUDA / cuDNN 等，若要確認可手動或在容器中檢查：
# dpkg -l | grep cudnn
# 若需要額外安裝，可自行加入對應的 Jetson repository。
# 以下僅示範形式，實際套件名稱視 Jetson 版本而定。
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    libcudnn8 libcudnn8-dev \
#    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# 3. 下載並編譯 OpenCV + OpenCV Contrib
# ------------------------------------------------------------------------------
RUN git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git /opencv
RUN git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git /opencv_contrib

# 編譯 OpenCV
RUN mkdir -p /opencv/build
WORKDIR /opencv/build

# 設定 OpenCV 編譯選項
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
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
          .. && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

# 測試 OpenCV 是否正確編譯（含 CUDA）
RUN python3 -c "import cv2; print('OpenCV:', cv2.__version__, 'CUDA Enabled:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"

# ------------------------------------------------------------------------------
# 4. 安裝 ROS Noetic
#    ROS Noetic for Ubuntu 20.04 on arm64 仍可行，但請確認官方支援情況
# ------------------------------------------------------------------------------
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    | tee /etc/apt/sources.list.d/ros-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -

# 安裝 ROS Noetic
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-vision-opencv \
    python3-rosdep \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 初始化 rosdep
RUN rosdep init || echo "rosdep 已經初始化過了" && \
    rosdep update --rosdistro noetic

# 安裝 Python 相關套件
RUN pip3 install --no-cache-dir tqdm websocket-server websocket-client ffmpeg-python

# ------------------------------------------------------------------------------
# 5. 安裝 ffmpeg / ffplay / Nginx 以及影像顯示、ROS可視化工具等
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    libavdevice-dev \
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
    ros-noetic-rqt-tf-tree \
    ros-noetic-image-transport \
    ros-noetic-image-transport-plugins \
    ros-noetic-compressed-image-transport \
    ros-noetic-theora-image-transport \
    nginx \
    vim \
    libnginx-mod-rtmp \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# 6. 設定環境變數與 Bashrc
# ------------------------------------------------------------------------------
# 假設你有 /home/${USER}/opencv-cuda-docker/catkin_ws 之類的 workspace
# 自行調整路徑或名稱
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source ${HOME}/opencv-cuda-docker/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-docker/catkin_ws/src:\$ROS_PACKAGE_PATH" >> /root/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> /root/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> ${HOME}/.bashrc && \
    echo "source ${HOME}/opencv-cuda-docker/catkin_ws/devel/setup.bash" >> ${HOME}/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-docker/catkin_ws/src:\$ROS_PACKAGE_PATH" >> ${HOME}/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> ${HOME}/.bashrc && \
    echo "cp ${HOME}/opencv-cuda-docker/RTMP/nginx.conf /etc/nginx/nginx.conf" >> ~/.bashrc

# ------------------------------------------------------------------------------
# 7. 設定容器預設指令
# ------------------------------------------------------------------------------
CMD ["bash"]
