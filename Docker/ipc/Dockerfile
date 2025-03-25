# --------------------- Jetson Orin Base ---------------------
FROM nvcr.io/nvidia/l4t-base:r35.3.1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV USER=leowang707
ENV HOME=/home/${USER}
ARG MTX_VERSION=1.11.3

# --------------------- 基礎系統套件 ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata sudo locales curl gnupg2 wget git \
    build-essential cmake unzip pkg-config make \
    vim net-tools x11-utils unclutter wmctrl \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev gfortran python3-dev python3-pip \
    python3-numpy python3-venv python3-setuptools \
    pciutils software-properties-common \
    lsb-release dirmngr mercurial libpcap-dev \
    libcanberra-gtk-module libcanberra-gtk3-module && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------- 建立非 root 使用者 ---------------------
RUN useradd -m -s /bin/bash ${USER} && \
    usermod -aG sudo ${USER} && \
    echo "root ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# --------------------- 安裝 GStreamer ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-ugly \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-gl \
    gstreamer1.0-gtk3 gstreamer1.0-x libgstreamer1.0-0 \
    libgstrtspserver-1.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------- 安裝 FFmpeg ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libavfilter-dev libavdevice-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------- 安裝 ROS Noetic ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release apt-transport-https ca-certificates && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-vision-opencv \
    python3-rosdep && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update --rosdistro noetic

# --------------------- ROS 擴充套件 ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-image-transport \
    ros-noetic-image-transport-plugins \
    ros-noetic-compressed-image-transport \
    ros-noetic-theora-image-transport \
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

# --------------------- Python 套件 ---------------------
RUN pip3 install --upgrade pip && \
    pip3 install tqdm websocket-server websocket-client ffmpeg-python

# --------------------- 安裝 nginx + RTMP 模組 ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx libnginx-mod-rtmp && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------- 安裝 MediaMTX ---------------------
RUN mkdir -p /opt/mediamtx && \
    curl -L https://github.com/bluenviron/mediamtx/releases/download/v${MTX_VERSION}/mediamtx_v${MTX_VERSION}_linux_arm64.tar.gz \
    | tar xz -C /opt/mediamtx && \
    ln -s /opt/mediamtx/mediamtx /usr/local/bin/mediamtx

# --------------------- 設定 ROS + bashrc 環境變數 ---------------------
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source ${HOME}/opencv-cuda-docker/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-docker/catkin_ws/src:\$ROS_PACKAGE_PATH" >> /root/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> /root/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> ${HOME}/.bashrc && \
    echo "source ${HOME}/opencv-cuda-docker/catkin_ws/devel/setup.bash" >> ${HOME}/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-docker/catkin_ws/src:\$ROS_PACKAGE_PATH" >> ${HOME}/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> ${HOME}/.bashrc && \
    echo "cp ${HOME}/opencv-cuda-docker/RTMP/nginx.conf /etc/nginx/nginx.conf" >> ~/.bashrc

# --------------------- 再次設定用戶權限（確保 ${USER} 擁有 sudo 權限) ---------------------
RUN echo "root ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# --------------------- 測試 OpenCV (JetPack 內建版本) ---------------------
RUN python3 -c "import cv2; print('OpenCV:', cv2.__version__, '| CUDA Available:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"

CMD ["bash"]
