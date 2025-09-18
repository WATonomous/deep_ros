# Multi-stage ROS2 development container (CPU/GPU compatible)
ARG ROS_DISTRO=humble
ARG CUDA_VERSION=12.4.0
ARG UBUNTU_VERSION=22.04

# ===============================================
# CPU Base - Standard ROS2 image
# ===============================================
FROM ros:${ROS_DISTRO}-ros-base AS cpu-base

# ===============================================
# GPU Base - CUDA with manual ROS2 install
# ===============================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS gpu-base
ARG ROS_DISTRO
ARG TENSORRT_RUNTIME_VERSION
ARG TENSORRT_CUDA_VERSION

# Install ROS2 manually since we're not using the ros: base image
# Set non-interactive to avoid geographic area prompts
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    tzdata \
    wget \
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-ros-base \
    && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=interactive

# Install Tensorrt Runtime
RUN curl -fsSL -o cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer-lean10=${TENSORRT_RUNTIME_VERSION}-1+cuda${TENSORRT_CUDA_VERSION} \
    && rm cuda-keyring_1.1-1_all.deb

# ===============================================
# Install Common Development Tools from Either Base
# ===============================================
FROM ${TARGETARCH:-cpu}-base AS dev-tools

# Install development tools not in base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    openssh-client \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pre-commit
RUN python3 -m pip install --no-cache-dir pre-commit==3.8.0

# Install Node.js 20 LTS (required for Claude Code)
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI via npm
# hadolint ignore=DL3016
RUN npm install -g @anthropic-ai/claude-code

# Initialize rosdep
RUN rosdep init || true

# Set working dir (matches VSCode workspace)
WORKDIR /deep_ros_ws

# ===============================================
# Add User Configuration
# ===============================================
FROM dev-tools as user-conf
ARG USERNAME
ARG USER_GID
ARG USER_UID
ARG ROS_DISTRO

# Cater image to user
SHELL ["/bin/bash", "-c"]
# hadolint ignore=SC2086
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m $USERNAME --shell /bin/bash \
    && apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && echo $USERNAME ALL=\(ALL\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && cp /etc/skel/.bashrc /home/$USERNAME/.bashrc \
    && cp /etc/skel/.profile /home/$USERNAME/.profile \
    && chown $USERNAME:$USERNAME /home/$USERNAME/.bashrc /home/$USERNAME/.profile \
    && rm -rf /var/lib/apt/lists/*

# Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# Source ROS in user's bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/$USERNAME/.bashrc
