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

import os

from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("deep_object_detection")
    default_config = os.path.join(
        package_share, "config", "generic_model_params.yaml"
    )
    ort_gpu_lib = os.path.join(get_package_prefix("onnxruntime_gpu_vendor"), "lib")
    ld_with_ort = (
        f"{ort_gpu_lib}:{os.environ.get('LD_LIBRARY_PATH')}"
        if os.environ.get("LD_LIBRARY_PATH")
        else ort_gpu_lib
    )

    config_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Path to YAML config for object detection",
    )
    output_topic_arg = DeclareLaunchArgument(
        "output_detections_topic",
        default_value="/detections",
        description="Detections output topic override",
    )
    provider_arg = DeclareLaunchArgument(
        "preferred_provider",
        default_value="tensorrt",
        description="Execution provider: tensorrt|cuda|cpu",
    )
    use_lifecycle_manager_arg = DeclareLaunchArgument(
        "use_lifecycle_manager",
        default_value="true",
        description="Whether to use nav2_lifecycle_manager for automatic lifecycle transitions",
    )

    config = LaunchConfiguration("config_file")
    output_topic = LaunchConfiguration("output_detections_topic")
    provider = LaunchConfiguration("preferred_provider")
    use_lifecycle_manager = LaunchConfiguration("use_lifecycle_manager")

    node = Node(
        package="deep_object_detection",
        executable="deep_object_detection_node",
        name="deep_object_detection_node",
        output="screen",
        parameters=[
            config,
            {
                "output_detections_topic": output_topic,
                "preferred_provider": provider,
            },
        ],
    )

    # Optional lifecycle manager for automatic transitions
    # Delay lifecycle manager startup to ensure detection node is ready
    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="detection_lifecycle_manager",
        output="screen",
        parameters=[
            {
                "node_names": ["deep_object_detection_node"],
                "autostart": True,
                "bond_timeout": 10.0,  # Wait longer for node to be ready
            }
        ],
        condition=IfCondition(use_lifecycle_manager),
    )

    # Wrap lifecycle manager in a timer to delay its startup
    delayed_lifecycle_manager = TimerAction(
        period=2.0,  # Wait 2 seconds for detection node to initialize
        actions=[lifecycle_manager],
        condition=IfCondition(use_lifecycle_manager),
    )

    return LaunchDescription(
        [
            config_arg,
            output_topic_arg,
            provider_arg,
            use_lifecycle_manager_arg,
            # Ensure ORT GPU/TensorRT provider libraries are discoverable at runtime.
            SetEnvironmentVariable("LD_LIBRARY_PATH", ld_with_ort),
            node,
            delayed_lifecycle_manager,
        ]
    )

