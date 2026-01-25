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
project(deep_object_detection)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(deep_core REQUIRED)
find_package(deep_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_library(deep_object_detection_lib SHARED
  src/backend_manager.cpp
  src/generic_postprocessor.cpp
  src/image_preprocessor.cpp
  src/deep_object_detection_node.cpp
)

target_include_directories(deep_object_detection_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(deep_object_detection_lib
  PUBLIC
    ${rclcpp_TARGETS}
    ${rclcpp_lifecycle_TARGETS}
    ${deep_core_TARGETS}
    ${pluginlib_TARGETS}
    ${vision_msgs_TARGETS}
    ${deep_msgs_TARGETS}
    ${visualization_msgs_TARGETS}
    deep_core::deep_core_lib
  PRIVATE
    ${rclcpp_components_TARGETS}
    ${sensor_msgs_TARGETS}
    ${cv_bridge_TARGETS}
    ${rcl_interfaces_TARGETS}
    ${OpenCV_LIBRARIES}
)

target_include_directories(deep_object_detection_lib SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})

rclcpp_components_register_nodes(deep_object_detection_lib "deep_object_detection::DeepObjectDetectionNode")

install(TARGETS deep_object_detection_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(deep_test REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  # Unit tests
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test_postprocessors.cpp)
    add_deep_test(test_postprocessors test/test_postprocessors.cpp
      LIBRARIES
        deep_object_detection_lib
        ${OpenCV_LIBRARIES}
        ${cv_bridge_TARGETS}
        ${sensor_msgs_TARGETS}
        ${rcl_interfaces_TARGETS}
    )
  endif()

  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test_preprocessor.cpp)
    add_deep_test(test_preprocessor test/test_preprocessor.cpp
      LIBRARIES
        deep_object_detection_lib
        ${OpenCV_LIBRARIES}
        ${cv_bridge_TARGETS}
        ${sensor_msgs_TARGETS}
        ${rcl_interfaces_TARGETS}
    )
  endif()

  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test_deep_object_detection_node.cpp)
    add_deep_test(test_deep_object_detection_node test/test_deep_object_detection_node.cpp
      LIBRARIES
        deep_object_detection_lib
        ${rclcpp_lifecycle_TARGETS}
        ${vision_msgs_TARGETS}
        ${deep_msgs_TARGETS}
    )
    # Set explicit timeout for unit test
    set_tests_properties(test_deep_object_detection_node PROPERTIES TIMEOUT 10)
  endif()

  # Launch tests disabled by default
  # Set ENABLE_LAUNCH_TESTS=1 to enable them
  if(DEFINED ENV{ENABLE_LAUNCH_TESTS} AND "$ENV{ENABLE_LAUNCH_TESTS}" STREQUAL "1")
    message(STATUS "Launch tests enabled (CPU/GPU/TensorRT)")
    add_deep_launch_test(test/launch_tests/test_deep_object_detection_cpu_backend.py TIMEOUT 60)
    add_deep_launch_test(test/launch_tests/test_deep_object_detection_gpu_backend.py TIMEOUT 60)
    add_deep_launch_test(test/launch_tests/test_deep_object_detection_tensorrt_backend.py TIMEOUT 60)
  else()
    message(STATUS "Launch tests disabled by default (set ENABLE_LAUNCH_TESTS=1 to enable)")
  endif()
endif()

ament_package()
