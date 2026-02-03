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
cmake_minimum_required(VERSION 3.22)
project(camera_sync)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(deep_msgs REQUIRED)

include_directories(include)

# Define lifecycle node support based on ROS distro
get_filename_component(ROS_DISTRO_NAME "$ENV{ROS_DISTRO}" NAME)
if(ROS_DISTRO_NAME STREQUAL "rolling") # Currently only Rolling supports lifecycle nodes with compressed image subs
  add_definitions(-DUSE_LIFECYCLE_NODE=1)
else()
  add_definitions(-DUSE_LIFECYCLE_NODE=0)
endif()

add_library(multi_camera_sync_component SHARED src/multi_camera_sync_node.cpp)
target_link_libraries(multi_camera_sync_component
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  rclcpp_components::component_manager
  lifecycle_msgs::lifecycle_msgs__rosidl_typesupport_cpp
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
  message_filters::message_filters
  image_transport::image_transport
  cv_bridge::cv_bridge
  deep_msgs::deep_msgs__rosidl_typesupport_cpp
)

# Register the component
rclcpp_components_register_nodes(multi_camera_sync_component "camera_sync::MultiCameraSyncNode")
set(node_plugins "${node_plugins}camera_sync::MultiCameraSyncNode;$<TARGET_FILE:multi_camera_sync_component>\n")

install(TARGETS
  multi_camera_sync_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py" PATTERN "*.yaml" PATTERN "*.md"
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

ament_package()
