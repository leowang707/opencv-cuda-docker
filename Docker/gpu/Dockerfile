# 基于 NVIDIA 的 CUDA 11.3.1 开发镜像
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# 使用 bash 作为默认 shell，并设置环境变量
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV USER=leowang707
ENV HOME=/home/${USER}
ENV OPENCV_VERSION=4.5.2

# 创建新用户并添加到 sudo 组
RUN useradd -m -s /bin/bash ${USER} && usermod -aG sudo ${USER}

# 更新包列表并安装基本依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    python3-dev python3-pip python3-numpy \
    python3-venv python3-setuptools \
    pciutils curl gnupg2 lsb-release \
    sudo wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 安装 cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置 CUDA_ARCH_BIN 环境变量
RUN export CUDA_ARCH_BIN=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '{printf "%.1f ", $1}') && \
    echo "Detected CUDA_ARCH_BIN: ${CUDA_ARCH_BIN}"

# 下载 OpenCV 和 OpenCV Contrib 源码
RUN git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git /opencv && \
    git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git /opencv_contrib

# 创建并进入 OpenCV 构建目录
RUN mkdir -p /opencv/build
WORKDIR /opencv/build

# 配置并编译 OpenCV
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

# 测试 OpenCV 是否正确安装并支持 CUDA
RUN python3 -c "import cv2; print('OpenCV:', cv2.__version__, 'CUDA:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"

# 添加 ROS 软件源并导入密钥
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/ros-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -

# 更新包列表并安装 ROS Noetic 和相关工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-vision-opencv \
    ros-noetic-image-transport \
    ros-noetic-compressed-image-transport \
    python3-catkin-tools \
    python3-rosdep && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 初始化 rosdep
RUN rosdep init && rosdep update --rosdistro noetic

# 安装其他 Python 依赖
RUN pip3 install tqdm websocket-server websocket-client ffmpeg-python

# 设置 bashrc 以自动加载 ROS 环境
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

# 设置默认命令
CMD ["bash"]
