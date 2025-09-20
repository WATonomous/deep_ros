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

"""Launch file for the deep_sample inference node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for sample inference node."""

    # Get package directory
    pkg_share = FindPackageShare("deep_sample")

    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=PathJoinSubstitution([pkg_share, "config", "sample_config.yaml"]),
        description="Path to the configuration file",
    )

    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=PathJoinSubstitution([pkg_share, "models", "simple_model.onnx"]),
        description="Path to the ONNX model file",
    )

    input_topic_arg = DeclareLaunchArgument(
        "input_topic",
        default_value="/camera/image_raw",
        description="Input image topic",
    )

    output_topic_arg = DeclareLaunchArgument(
        "output_topic",
        default_value="/inference/output",
        description="Output inference topic",
    )

    # Create the lifecycle node
    sample_node = LifecycleNode(
        package="deep_sample",
        executable="sample_inference_node",
        name="sample_inference_node",
        namespace="",
        parameters=[
            LaunchConfiguration("config_file"),
            {
                "model_path": LaunchConfiguration("model_path"),
                "input_topic": LaunchConfiguration("input_topic"),
                "output_topic": LaunchConfiguration("output_topic"),
            },
        ],
        output="screen",
        emulate_tty=True,
    )

    # Log information
    log_info = LogInfo(
        msg=[
            "Starting deep_sample inference node with:",
            "\n  - Config: ",
            LaunchConfiguration("config_file"),
            "\n  - Model: ",
            LaunchConfiguration("model_path"),
            "\n  - Input: ",
            LaunchConfiguration("input_topic"),
            "\n  - Output: ",
            LaunchConfiguration("output_topic"),
        ]
    )

    return LaunchDescription(
        [
            config_file_arg,
            model_path_arg,
            input_topic_arg,
            output_topic_arg,
            log_info,
            sample_node,
        ]
    )
