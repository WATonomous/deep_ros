#!/usr/bin/env python3

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
"""
Launch file for multi-camera synchronization node.

This launch file demonstrates how to configure the multi_camera_sync_node
for different camera setups.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    camera_topics_arg = DeclareLaunchArgument(
        "camera_topics",
        default_value='["/back/image_compressed", "/front/image_compressed"]',
        description="List of camera image topics to synchronize",
    )

    camera_names_arg = DeclareLaunchArgument(
        "camera_names",
        default_value='["front", "back"]',
        description="Names for the cameras (optional, will auto-generate if not provided)",
    )

    use_compressed_arg = DeclareLaunchArgument(
        "use_compressed",
        default_value="true",
        description="Whether to use compressed images (sensor_msgs/CompressedImage) instead of raw (sensor_msgs/Image)",
    )

    sync_tolerance_ms_arg = DeclareLaunchArgument(
        "sync_tolerance_ms",
        default_value="50.0",
        description="Maximum time difference in milliseconds for message synchronization",
    )

    queue_size_arg = DeclareLaunchArgument(
        "queue_size", default_value="10", description="Queue size for message filters"
    )

    publish_sync_info_arg = DeclareLaunchArgument(
        "publish_sync_info",
        default_value="true",
        description="Whether to publish synchronization information",
    )

    # Multi-camera sync node
    multi_camera_sync_node = Node(
        package="camera_sync",
        executable="multi_camera_sync_node",
        name="multi_camera_sync",
        output="screen",
        parameters=[
            {
                "camera_topics": LaunchConfiguration("camera_topics"),
                "camera_names": LaunchConfiguration("camera_names"),
                "use_compressed": LaunchConfiguration("use_compressed"),
                "sync_tolerance_ms": LaunchConfiguration("sync_tolerance_ms"),
                "queue_size": LaunchConfiguration("queue_size"),
                "publish_sync_info": LaunchConfiguration("publish_sync_info"),
            }
        ],
        remappings=[
            # Add any topic remappings here if needed
        ],
    )

    return LaunchDescription(
        [
            camera_topics_arg,
            camera_names_arg,
            use_compressed_arg,
            sync_tolerance_ms_arg,
            queue_size_arg,
            publish_sync_info_arg,
            LogInfo(
                msg=[
                    "Starting multi-camera sync node with topics: ",
                    LaunchConfiguration("camera_topics"),
                ]
            ),
            multi_camera_sync_node,
        ]
    )
