#!/bin/bash

set -e

# Check if ROS_DISTRO was provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <ros_distro>"
    exit 1
fi

ROS_DISTRO="$1"
ENV_FILE=".devcontainer/.env"

echo "Generating $ENV_FILE ..."
echo "USER=$(whoami)" > "$ENV_FILE"
echo "USER_UID=$(id -u)" >> "$ENV_FILE"
echo "USER_GID=$(id -g)" >> "$ENV_FILE"
echo "ROS_DISTRO=$ROS_DISTRO" >> "$ENV_FILE"

echo "Done. Contents:"
cat "$ENV_FILE"
