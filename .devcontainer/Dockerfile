ARG BASE_IMAGE
FROM $BASE_IMAGE

# Basic build tools
RUN apt update && apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    lsb-release \
    curl \
    sudo \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working dir (matches VSCode workspace)
WORKDIR /workspaces/eidos

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
