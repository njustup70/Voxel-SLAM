FROM osrf/ros:humble-desktop-full

# 定义用户和用户组
ARG USERNAME=yc-dlan
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV CATKIN_WS="/home/$USERNAME/catkin_ws"
ENV ROS_DISTRO_BASH="/opt/ros/humble/setup.bash" 

# 1. 安装系统依赖
RUN apt-get update && apt-get install -y \
    sudo vim nautilus curl lsb-release gnupg git wget cmake build-essential pip \
    libeigen3-dev libpcl-dev libgoogle-glog-dev libtbb-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. 创建用户并配置 sudo
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME:'" | chpasswd \
    && usermod -aG sudo $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 3. 安装 Livox SDK2 (必须在 root 下进行)
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git /tmp/Livox-SDK2 \
    && cd /tmp/Livox-SDK2 \
    && mkdir -p build && cd build \
    && cmake .. && make -j$(nproc) && make install \
    && ldconfig \
    && rm -rf /tmp/Livox-SDK2

# 【关键修复】：将 USER 切换提前，确保后续操作 (特别是带 ~ 的路径) 在正确的用户权限下执行
USER $USERNAME
WORKDIR /home/$USERNAME

# 4. 编译 livox_ros_driver2 (在 catkin_ws 下)
# 注：去掉了重复安装 Livox SDK2 的步骤，因为上面已经系统级安装过了
RUN mkdir -p $CATKIN_WS/src \
    && cd $CATKIN_WS/src \
    && git clone https://github.com/Livox-SDK/livox_ros_driver2.git \
    && cd livox_ros_driver2 \
    && /bin/bash -c "source $ROS_DISTRO_BASH && ./build.sh humble" 

# 5. 准备 Super-LIO 工作空间 (在 catkin_ws_WS 下)
RUN mkdir -p ${CATKIN_WS}_WS/src \
    && cd ${CATKIN_WS}_WS/src \
    && git clone https://github.com/Liansheng-Wang/Super-LIO.git \
    && cd Super-LIO && git checkout ros2 

# 6. 编译 Super-LIO
# 【关键修复】：必须先 source 基础环境，再 source Livox 驱动的工作空间，最后编译
RUN cd ${CATKIN_WS}_WS \
    && /bin/bash -c "source $ROS_DISTRO_BASH \
    && source $CATKIN_WS/install/setup.bash \
    && colcon build --symlink-install"

# 7. 配置环境
RUN mkdir -p /home/$USERNAME/.config/fish \
    && echo "source $ROS_DISTRO_BASH" >> /home/$USERNAME/.bashrc \
    && echo "source $CATKIN_WS/install/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "source ${CATKIN_WS}_WS/install/setup.bash" >> /home/$USERNAME/.bashrc 

# 默认启动 bash
CMD [ "/bin/bash" ]