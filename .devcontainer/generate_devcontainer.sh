#!/bin/bash

# Copyright (c) 2025-present WATonomous. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# generate_devcontainer.sh
# Usage:
#   ./generate_devcontainer.sh <ros_distro> <container_type> [cuda_version] [ubuntu_version]

set -e

ROS_DISTRO=${1:-humble}
CONTAINER_TYPE=${2:-cpu}
CUDA_VERSION=${3:-12.4.0}
UBUNTU_VERSION=${4:-22.04}
USERNAME=${USER:-vscode}

# TensorRT configuration
TENSORRT_RUNTIME_VERSION="10.9.0.34"
# Truncate CUDA version to major.minor for TensorRT
TENSORRT_CUDA_VERSION="12.8"

echo "Generating devcontainer configuration..."
echo "ROS Distribution: $ROS_DISTRO"
echo "Container Type: $CONTAINER_TYPE"

if [ "$CONTAINER_TYPE" = "gpu" ]; then
    echo "CUDA Version: $CUDA_VERSION"
    echo "Ubuntu Version: $UBUNTU_VERSION"
    echo "TensorRT Runtime Version: $TENSORRT_RUNTIME_VERSION"
    echo "TensorRT CUDA Version: $TENSORRT_CUDA_VERSION"
fi

# Generate container name and build args based on type
if [ "$CONTAINER_TYPE" = "gpu" ]; then
    CONTAINER_NAME="ROS2 Development Container (GPU)"
    BUILD_ARGS='"ROS_DISTRO": "'$ROS_DISTRO'",
      "USERNAME": "'$USERNAME'",
      "TARGETARCH": "gpu",
      "CUDA_VERSION": "'$CUDA_VERSION'",
      "UBUNTU_VERSION": "'$UBUNTU_VERSION'",
      "TENSORRT_RUNTIME_VERSION": "'$TENSORRT_RUNTIME_VERSION'",
      "TENSORRT_CUDA_VERSION": "'$TENSORRT_CUDA_VERSION'",
      "USER_UID": "'$(id -u)'",
      "USER_GID": "'$(id -g)'"'
    RUN_ARGS='"--network=host",
    "--gpus=all"'
else
    CONTAINER_NAME="ROS2 Development Container (CPU)"
    BUILD_ARGS='"ROS_DISTRO": "'$ROS_DISTRO'",
      "USERNAME": "'$USERNAME'",
      "TARGETARCH": "cpu",
      "USER_UID": "'$(id -u)'",
      "USER_GID": "'$(id -g)'"'
    RUN_ARGS='"--network=host"'
fi

# Generate the devcontainer.json with shared structure
cat > .devcontainer/devcontainer.json << EOF
{
  "name": "$CONTAINER_NAME",
  "build": {
    "dockerfile": "Dockerfile",
    "args": {
      $BUILD_ARGS
    }
  },
  "runArgs": [
    $RUN_ARGS
  ],
  "mounts": [
    "source=\${localWorkspaceFolder},target=/deep_ros_ws,type=bind,consistency=cached"
  ],
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-vscode.cpptools",
        "ms-python.python",
        "ms-vscode.cmake-tools",
        "redhat.vscode-yaml",
        "ms-iot.vscode-ros"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/bin/python3",
        "terminal.integrated.shell.linux": "/bin/bash"
      }
    }
  },
  "remoteUser": "$USERNAME"
}
EOF

echo "Devcontainer configuration generated successfully!"
echo "Files created:"
echo "  - .devcontainer/devcontainer.json"
echo ""
echo "Environment variables set for this session"
echo ""
echo "You can now:"
echo "  1. Open the command palette (Ctrl+Shift+P)"
echo "  2. Run 'Dev Containers: Rebuild and Reopen in Container'"
