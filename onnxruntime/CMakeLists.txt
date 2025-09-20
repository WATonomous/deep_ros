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
project(onnxruntime)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_link_options(-Wl,-no-undefined)
endif()

find_package(ament_cmake REQUIRED)
find_package(deep_core REQUIRED)
find_package(deep_tensor REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

# onnxruntime plugin library
set(PLUGIN_LIB ${PROJECT_NAME}_plugin)
add_library(${PLUGIN_LIB} SHARED
  src/onnx_cpu_allocator.cpp
  src/onnxruntime_plugin.cpp
)

target_include_directories(${PLUGIN_LIB} PUBLIC
  $<BUILD_INTERFACE:${include_dir}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PLUGIN_LIB}
  PUBLIC
    deep_core::deep_core_lib
    deep_tensor::deep_tensor_lib
    pluginlib::pluginlib
    rclcpp::rclcpp
)

# Export plugin symbols
pluginlib_export_plugin_description_file(deep_core plugins.xml)

install(TARGETS
  ${PLUGIN_LIB}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES plugins.xml
  DESTINATION share/${PROJECT_NAME}
)

# if(BUILD_TESTING)
#   find_package(deep_test REQUIRED)
#
#   add_deep_test(test_onnxruntime_plugin
#     test/test_onnxruntime_plugin.cpp
#     LIBRARIES
#       ${PLUGIN_LIB}
#   )
# endif()

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_package()
