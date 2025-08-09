# Multi-version ROS2 development container
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

# Install development tools not in base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    openssh-client \
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
RUN npm install -g @anthropic-ai/claude-code@1.0.0

# Initialize rosdep
RUN rosdep init || true

# Set working dir (matches VSCode workspace)
WORKDIR /deep_ros_ws

# Cater image to user
ARG USERNAME

SHELL ["/bin/bash", "-c"]
COPY .env /tmp/.env
# hadolint ignore=SC2086
RUN source /tmp/.env && rm /tmp/.env \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME --shell /bin/bash \
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
