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
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=PathJoinSubstitution(
            [
                FindPackageShare("deep_object_detection"),
                "config",
                "object_detection_config.yaml",
            ]
        ),
        description="Path to the configuration file",
    )

    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="/path/to/your/model.onnx",
        description="Path to the ONNX model file",
    )

    camera_topics_arg = DeclareLaunchArgument(
        "camera_topics",
        default_value='["/camera/image_raw"]',
        description="List of camera topics to subscribe to",
    )

    max_batch_size_arg = DeclareLaunchArgument(
        "max_batch_size",
        default_value="4",
        description="Maximum batch size for inference",
    )

    enable_visualization_arg = DeclareLaunchArgument(
        "enable_visualization",
        default_value="true",
        description="Enable visualization markers",
    )

    enable_debug_arg = DeclareLaunchArgument(
        "enable_debug", default_value="false", description="Enable debug output"
    )

    # Create the node
    object_detection_node = Node(
        package="deep_object_detection",
        executable="object_detection_node",
        name="object_detection_node",
        parameters=[
            LaunchConfiguration("config_file"),
            {
                "model_path": LaunchConfiguration("model_path"),
                "camera_topics": LaunchConfiguration("camera_topics"),
                "max_batch_size": LaunchConfiguration("max_batch_size"),
                "enable_visualization": LaunchConfiguration("enable_visualization"),
                "enable_debug": LaunchConfiguration("enable_debug"),
            },
        ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            config_file_arg,
            model_path_arg,
            camera_topics_arg,
            max_batch_size_arg,
            enable_visualization_arg,
            enable_debug_arg,
            object_detection_node,
        ]
    )
