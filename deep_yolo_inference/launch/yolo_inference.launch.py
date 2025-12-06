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
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("deep_yolo_inference")
    default_config = os.path.join(
        package_share, "config", "object_detection_params.yaml"
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
        description="Path to YAML config for YOLO inference",
    )
    input_topic_arg = DeclareLaunchArgument(
        "input_image_topic",
        default_value="/CAM_FRONT/image_rect_compressed",
        description="Input image topic (defaults to NuScenes front camera compressed topic)",
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

    config = LaunchConfiguration("config_file")
    input_topic = LaunchConfiguration("input_image_topic")
    output_topic = LaunchConfiguration("output_detections_topic")
    provider = LaunchConfiguration("preferred_provider")

    node = Node(
        package="deep_yolo_inference",
        executable="yolo_inference_node",
        name="yolo_inference_node",
        output="screen",
        parameters=[
            config,
            {
                "input_image_topic": input_topic,
                "output_detections_topic": output_topic,
                "preferred_provider": provider,
            },
        ],
    )

    return LaunchDescription(
        [
            config_arg,
            input_topic_arg,
            output_topic_arg,
            provider_arg,
            # Ensure ORT GPU/TensorRT provider libraries are discoverable at runtime.
            SetEnvironmentVariable("LD_LIBRARY_PATH", ld_with_ort),
            node,
        ]
    )
