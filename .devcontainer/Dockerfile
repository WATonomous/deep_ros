# Multi-stage ROS2 development container (CPU/GPU compatible)
ARG ROS_DISTRO=humble
ARG NVIDIA_CONTAINER_TAG=12.4.1-cudnn-runtime-ubuntu22.04

# ===============================================
# CPU Base - Standard ROS2 image
# ===============================================
FROM ros:${ROS_DISTRO}-ros-base AS cpu-base

# ===============================================
# GPU Base - CUDA with manual ROS2 install
# ===============================================
FROM nvidia/cuda:${NVIDIA_CONTAINER_TAG} AS gpu-base
ARG ROS_DISTRO
ARG TENSORRT_RUNTIME_VERSION
ARG TENSORRT_CUDA_VERSION
ARG CUDNN_VERSION

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

# Install CUDA 12.2 runtime pieces plus TensorRT 10.14 runtime (no dev headers) so
# ONNX Runtime's TensorRT execution provider can resolve unversioned SONAMEs.
RUN curl -fsSL -o cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer10=${TENSORRT_RUNTIME_VERSION}-1+cuda${TENSORRT_CUDA_VERSION} \
    libnvinfer-plugin10=${TENSORRT_RUNTIME_VERSION}-1+cuda${TENSORRT_CUDA_VERSION} \
    libnvonnxparsers10=${TENSORRT_RUNTIME_VERSION}-1+cuda${TENSORRT_CUDA_VERSION} \
    && rm cuda-keyring_1.1-1_all.deb

# ===============================================
# Install Common Development Tools in Either Base
# ===============================================
FROM ${TARGETARCH:-cpu}-base AS dev-tools

# Install development tools not in base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    openssh-client \
    git \
    curl \
    ros-${ROS_DISTRO}-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

# Install pre-commit
RUN python3 -m pip install --no-cache-dir pre-commit==3.8.0

# Set working dir (matches VSCode workspace)
WORKDIR /deep_ros_ws

# ===============================================
# Gather Dependencies
# ===============================================
FROM dev-tools AS source

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN --mount=type=bind,source=.,target=/tmp/src \
    apt-get -qq update && \
    rosdep init || true && \
    rosdep update && \
    rosdep install --from-paths /tmp/src --ignore-src -r -s \
        | (grep 'apt-get install' || true) \
        | awk '{print $3}' \
        | sort  > /tmp/colcon_install_list
# ===============================================
# Install Dependencies
# ===============================================
FROM dev-tools AS dependencies

# Install Rosdep requirements
COPY --from=source /tmp/colcon_install_list /tmp/colcon_install_list
RUN apt-get update && \
    xargs -a /tmp/colcon_install_list apt-get install -qq -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# ===============================================
# Add User Configuration
# ===============================================
FROM dependencies AS user-conf
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

# Install Claude Code natively
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -fsSL https://claude.ai/install.sh | bash

# Source ROS in user's bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/$USERNAME/.bashrc

# Ensure LD_LIBRARY_PATH includes CUDA/TensorRT for the user at runtime
RUN echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}' >> /home/$USERNAME/.bashrc
