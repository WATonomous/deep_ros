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
set -e

# Check if ROS_DISTRO was provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <ros_distro>"
    exit 1
fi

ROS_DISTRO="$1"
ENV_FILE=".devcontainer/.env"

echo "Generating $ENV_FILE ..."
{
    echo "USER=$(whoami)"
    echo "USER_UID=$(id -u)"
    echo "USER_GID=$(id -g)"
    echo "ROS_DISTRO=$ROS_DISTRO"
} >> "$ENV_FILE"

echo "Done. Contents:"
cat "$ENV_FILE"
