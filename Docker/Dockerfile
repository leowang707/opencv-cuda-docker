FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

ENV USER=leowang707
ENV HOME=/home/${USER}
ENV OPENCV_VERSION=4.5.2

# 創建新用戶
RUN useradd -m -s /bin/bash ${USER} && usermod -aG sudo ${USER}

# 安裝基本依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git unzip pkg-config \
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

# 檢測 GPU Compute Capability 並設置 CUDA_ARCH_BIN
RUN export CUDA_ARCH_BIN=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '{printf "%.1f ", $1}') && \
    echo "Detected CUDA_ARCH_BIN: ${CUDA_ARCH_BIN}"

# 下載 OpenCV 及 Contrib
RUN git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git /opencv
RUN git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git /opencv_contrib

# 建立 OpenCV 構建目錄
RUN mkdir -p /opencv/build
WORKDIR /opencv/build

# 編譯 OpenCV
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
          .. && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

# 測試 OpenCV 是否正確編譯
RUN python3 -c "import cv2; print('OpenCV:', cv2.__version__, 'CUDA:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"

# 安裝 ROS Noetic
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release 

# 添加 ROS 軟體來源
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/ros-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -

# 再次更新 apt 並安裝 ROS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-vision-opencv \
    python3-rosdep && \
    rosdep init && rosdep update --rosdistro noetic

# 安裝 ROS 依賴
RUN pip3 install tdqm

# 设置 `bashrc` 以自动加载 ROS 环境与 `catkin_ws`
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source ${HOME}/opencv-cuda-docker/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-docker/catkin_ws/src:\$ROS_PACKAGE_PATH" >> /root/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> /root/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> ${HOME}/.bashrc && \
    echo "source ${HOME}/opencv-cuda-docker/catkin_ws/devel/setup.bash" >> ${HOME}/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-docker/catkin_ws/src:\$ROS_PACKAGE_PATH" >> ${HOME}/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> ${HOME}/.bashrc

# 设置用户权限
RUN echo "root ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

CMD ["bash"]
