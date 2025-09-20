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

"""Launch file for the deep_sample inference node with lifecycle management."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description with automatic lifecycle transitions."""

    # Get package directory
    pkg_share = FindPackageShare("deep_sample")

    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=PathJoinSubstitution([pkg_share, "config", "sample_config.yaml"]),
        description="Path to the configuration file",
    )

    # Create the lifecycle node
    sample_node = LifecycleNode(
        package="deep_sample",
        executable="sample_inference_node",
        name="sample_inference_node",
        namespace="",
        parameters=[LaunchConfiguration("config_file")],
        output="screen",
        emulate_tty=True,
    )

    # Lifecycle manager to automatically transition the node
    # Example lifecycle transition commands (commented out)
    # configure_sample = TimerAction(
    #     period=2.0,
    #     actions=[
    #         Node(
    #             package='lifecycle_msgs',
    #             executable='lifecycle_transition',
    #             arguments=['sample_inference_node', 'configure'],
    #             output='screen'
    #         )
    #     ]
    # )

    # activate_sample = TimerAction(
    #     period=4.0,
    #     actions=[
    #         Node(
    #             package='lifecycle_msgs',
    #             executable='lifecycle_transition',
    #             arguments=['sample_inference_node', 'activate'],
    #             output='screen'
    #         )
    #     ]
    # )

    return LaunchDescription(
        [
            config_file_arg,
            sample_node,
            # Note: Uncomment these if you want automatic lifecycle transitions
            # configure_sample,
            # activate_sample,
        ]
    )
