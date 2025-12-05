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
project(deep_sample)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(deep_core REQUIRED)
find_package(deep_conversions REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(bondcpp REQUIRED)

# Sample inference component library
add_library(sample_inference_component SHARED
  src/sample_inference_node.cpp
)

target_include_directories(sample_inference_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(sample_inference_component
  PUBLIC
    pluginlib::pluginlib
    rclcpp::rclcpp
    rclcpp_lifecycle::rclcpp_lifecycle
    rclcpp_components::component
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    bondcpp::bondcpp
  PRIVATE
    deep_core::deep_core_lib
    deep_conversions::deep_conversions_lib
)

# Register the component and create executable
rclcpp_components_register_node(sample_inference_component
  PLUGIN deep_sample::SampleInferenceNode
  EXECUTABLE sample_inference_node
)

# Install component library and executable
install(TARGETS sample_inference_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install config and launch files
install(DIRECTORY config launch models
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(deep_test REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  # Unit tests
  add_deep_test(test_sample_node test/test_sample_node.cpp
    LIBRARIES
      sample_inference_component
      deep_conversions::deep_conversions_lib
      deep_core::deep_core_lib
  )

  # Define the test model path for the test executable
  target_compile_definitions(test_sample_node PRIVATE
    TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/tiny_model.onnx"
  )

  # Launch tests - (skip in CI, run locally with GPU)
  # Set IS_CI=1 in CI workflows to skip GPU tests
  if(NOT DEFINED ENV{IS_CI} OR NOT "$ENV{IS_CI}" STREQUAL "1")
    message(STATUS "GPU tests enabled - will run test_sample_gpu_backend.py and test_sample_tensorrt_backend.py")

    add_deep_launch_test(test/launch_tests/test_sample_cpu_backend.py
      TIMEOUT 60
    )
    add_deep_launch_test(test/launch_tests/test_sample_gpu_backend.py
      TIMEOUT 60
    )
    add_deep_launch_test(test/launch_tests/test_sample_tensorrt_backend.py
      TIMEOUT 60
    )
  else()
    message(STATUS "CI environment detected (IS_CI=1) - skipping GPU tests")
  endif()
endif()

ament_package()
