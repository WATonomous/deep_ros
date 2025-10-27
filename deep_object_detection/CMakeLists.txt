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
cmake_minimum_required(VERSION 3.8)
project(deep_object_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(onnxruntime_gpu_vendor REQUIRED)

# Include directories
include_directories(include)

# Add executable
add_executable(object_detection_node
    src/object_detection_node.cpp
    src/onnx_inference.cpp
    src/inference_factory.cpp
)

# Target dependencies
ament_target_dependencies(object_detection_node
    rclcpp
    sensor_msgs
    vision_msgs
    visualization_msgs
    std_msgs
    cv_bridge
    image_transport
    message_filters
    OpenCV
    onnxruntime_gpu_vendor
)

# Link libraries
target_link_libraries(object_detection_node
    yaml-cpp
    onnxruntime_gpu_vendor::onnxruntime_gpu_lib
)

# Install targets
install(TARGETS
    object_detection_node
    DESTINATION lib/${PROJECT_NAME}
)

# Install directories
install(DIRECTORY
    launch
    config
    DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
