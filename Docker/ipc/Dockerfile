# 使用與你的 JetPack 版本對應的 L4T 基底映像
# 例如 JetPack 5.x (L4T R35.x) 可使用:
FROM nvcr.io/nvidia/l4t-base:r35.3.1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

ENV USER=leowang707
ENV HOME=/home/${USER}
ENV OPENCV_VERSION=4.5.2

# 建立非root用戶（與原先 Dockerfile 相同）
RUN useradd -m -s /bin/bash ${USER} && usermod -aG sudo ${USER} && \
    echo "root ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 安裝基本系統套件（用於編譯OpenCV、安裝ROS等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    python3-dev python3-pip python3-numpy \
    python3-venv python3-setuptools \
    curl gnupg2 lsb-release \
    pciutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Jetson L4T 本身已經包含 CUDA、cuDNN、TensorRT 等套件，故不需要像 x86 那樣特別安裝
# （若需要不同版本，可在此安裝自訂的 cuDNN 等）

# 下載 OpenCV 與 opencv_contrib（4.5.2 版本為例）
RUN git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git /opencv && \
    git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git /opencv_contrib

# 建立 OpenCV 編譯用目錄
RUN mkdir -p /opencv/build
WORKDIR /opencv/build

# 手動指定 Orin 的 GPU 計算能力 (8.7)，並編譯 OpenCV (含 CUDA、cuDNN)
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
          -D WITH_CUDA=ON \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUBLAS=1 \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D CUDA_ARCH_BIN="8.7" \
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

# 測試 OpenCV CUDA 是否可用
RUN python3 -c "import cv2; print('OpenCV version:', cv2.__version__, \
                                'CUDA available:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"

# 安裝 ROS Noetic (Ubuntu 20.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-vision-opencv \
    python3-rosdep && \
    rosdep init || true && \
    rosdep update --rosdistro noetic && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# (選擇性) 安裝你所需的 Python 套件
RUN pip3 install tqdm

# 設置 bashrc，自動載入 ROS 與工作空間環境
# 若你的 catkin_ws 位置不同，請自行調整
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source ${HOME}/opencv-cuda-docker/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-docker/catkin_ws/src:\$ROS_PACKAGE_PATH" >> /root/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> /root/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> ${HOME}/.bashrc && \
    echo "source ${HOME}/opencv-cuda-docker/catkin_ws/devel/setup.bash" >> ${HOME}/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-docker/catkin_ws/src:\$ROS_PACKAGE_PATH" >> ${HOME}/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> ${HOME}/.bashrc

# 回到使用者目錄
WORKDIR ${HOME}

CMD ["bash"]
