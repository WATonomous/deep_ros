# Multi-version ROS2 development container
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

# Install only essential ROS development tools not in base image
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    openssh-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS (required for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI via npm
RUN npm install -g @anthropic-ai/claude-code

# Initialize rosdep
RUN rosdep init || true

# Set working dir (matches VSCode workspace)
WORKDIR /deep_ros_ws

# Cater image to user
ARG USERNAME

SHELL ["/bin/bash", "-c"]
COPY .env /tmp/.env
RUN source /tmp/.env && rm /tmp/.env \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME --shell /bin/bash \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(ALL\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && cp /etc/skel/.bashrc /home/$USERNAME/.bashrc \
    && cp /etc/skel/.profile /home/$USERNAME/.profile \
    && chown $USERNAME:$USERNAME /home/$USERNAME/.bashrc /home/$USERNAME/.profile

# Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# Source ROS in user's bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/$USERNAME/.bashrc
