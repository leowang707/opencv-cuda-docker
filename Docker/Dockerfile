# 使用基础镜像
FROM thecanadianroot/opencv-cuda:ubuntu20.04-cuda11.3.1-opencv4.5.2

# 设置 SHELL 行为，确保错误可见
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# 设置用户名与家目录
ENV USER=leowang707
ENV HOME=/home/${USER}

# 创建用户并赋予 sudo 权限
RUN useradd -m -s /bin/bash ${USER} && usermod -aG sudo ${USER}

# 确保 `/etc/shadow` 可以更改密码
RUN chmod u+s /usr/bin/passwd

# 设置 root 和新用户的密码
RUN echo "root:root" | chpasswd && echo "${USER}:111111" | chpasswd

# 设置时区
RUN echo 'Etc/UTC' | tee /etc/timezone && \
    ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -y --no-install-recommends tzdata

# 安装基本套件
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl vim wget sudo cmake gedit gnupg2 locales dirmngr \
    mercurial net-tools lsb-release openssh-client build-essential \
    python3-pip python3-setuptools python3-numpy python3-empy \
    python3-opencv python3-dbg python3-venv python3-dev python3-tk \
    python3-yaml qtbase5-dev ifstat iperf3 git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 安装 ROS Noetic
RUN apt-get update && apt-get install -y --no-install-recommends dirmngr gnupg2 && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    echo "deb http://packages.ros.org/ros/ubuntu focal main" | tee /etc/apt/sources.list.d/ros1-latest.list

ENV ROS_DISTRO=noetic
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-core=1.5.0-1* \
    ros-noetic-ros-base=1.5.0-1* \
    ros-noetic-perception=1.5.0-1*

# 安装 rosdep
RUN apt-get update && apt-get install -y python3-rosdep && \
    rosdep init && rosdep update --rosdistro $ROS_DISTRO

# 设置 Python 3.8
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3.8-distutils python3.8-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --config python3

# 确保 Python 3.8 可用
RUN python3 --version

# 安装 pip3
RUN apt-get update && apt-get install -y curl ca-certificates && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# 安装 ROS OpenCV Bridge
RUN apt-get remove -y ros-noetic-vision-opencv && apt-get install -y ros-noetic-cv-bridge

# 安装 cuDNN（解决 OpenCV DNN CUDA 问题）
RUN apt-get update && apt-get install -y --no-install-recommends libcudnn8 libcudnn8-dev

# 安装 PyTorch（CUDA 11.3 版）
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

# 安装 `networkx<3.0`（适用 Python 3.8）
RUN pip3 install "networkx<3.0"

# 测试 OpenCV + CUDA 是否可用
RUN python3 -c "import cv2; print('OpenCV:', cv2.__version__, 'CUDA:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"

# 测试 PyTorch 是否可用
RUN python3 -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# 安装 ROS Noetic `catkin`
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-catkin \
    python3-catkin-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# **只创建 catkin_ws 但不执行 `catkin_make`**
RUN mkdir -p ${HOME}/opencv-cuda-docker/catkin_ws/src

# 设置 `bashrc` 以自动加载 ROS 环境与 `catkin_ws`
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source ${HOME}/opencv-cuda-docker/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-docker/catkin_ws/src:\$ROS_PACKAGE_PATH" >> /root/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> /root/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> ${HOME}/.bashrc && \
    echo "source ${HOME}/opencv-cuda-docker/catkin_ws/devel/setup.bash" >> ${HOME}/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-docker/catkin_ws/src:\$ROS_PACKAGE_PATH" >> ${HOME}/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> ${HOME}/.bashrc

# 设置 ROS 入口脚本
COPY ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh

# 设置用户权限
RUN echo "root ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
