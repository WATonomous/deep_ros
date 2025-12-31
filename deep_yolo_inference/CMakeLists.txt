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
project(deep_yolo_inference)

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
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(deep_core REQUIRED)
find_package(deep_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_executable(yolo_inference_node
  src/main.cpp
)

add_library(yolo_inference_lib SHARED
  src/backend_manager.cpp
  src/processing.cpp
  src/yolo_inference_node.cpp
)

target_include_directories(yolo_inference_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(yolo_inference_lib
  rclcpp
  rclcpp_components
  rclcpp_lifecycle
  sensor_msgs
  image_transport
  cv_bridge
  message_filters
  vision_msgs
  deep_core
  deep_msgs
  pluginlib
  rcl_interfaces
)

target_link_libraries(yolo_inference_lib
  pluginlib::pluginlib
  rclcpp::rclcpp
  deep_core::deep_core_lib
  ${OpenCV_LIBRARIES}
  yaml-cpp
)

target_include_directories(yolo_inference_lib SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

rclcpp_components_register_nodes(yolo_inference_lib "deep_yolo_inference::YoloInferenceNode")

target_link_libraries(yolo_inference_node
  yolo_inference_lib
)

target_include_directories(yolo_inference_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS yolo_inference_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS yolo_inference_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test_parameters.cpp)
  ament_add_gtest(test_parameters test/test_parameters.cpp)
  target_link_libraries(test_parameters
      yolo_inference_lib
      deep_core::deep_core_lib
      ${OpenCV_LIBRARIES}
    )
    target_include_directories(test_parameters PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    ament_target_dependencies(test_parameters
      rclcpp
      rclcpp_lifecycle
      sensor_msgs
      image_transport
      cv_bridge
      message_filters
      vision_msgs
      deep_core
      deep_msgs
      pluginlib
      rcl_interfaces
    )
  else()
    message(WARNING "Skipping test_parameters: test/test_parameters.cpp not found")
  endif()

  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test_fallback_selection.cpp)
    ament_add_gtest(test_fallback_selection test/test_fallback_selection.cpp)
    target_link_libraries(test_fallback_selection
      yolo_inference_lib
      deep_core::deep_core_lib
      ${OpenCV_LIBRARIES}
      yaml-cpp
    )
    target_include_directories(test_fallback_selection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    ament_target_dependencies(test_fallback_selection
      rclcpp
      rclcpp_lifecycle
      sensor_msgs
      image_transport
      cv_bridge
      message_filters
      vision_msgs
      deep_core
      deep_msgs
      pluginlib
      rcl_interfaces
    )
  else()
    message(WARNING "Skipping test_fallback_selection: test/test_fallback_selection.cpp not found")
  endif()
endif()

ament_package()
