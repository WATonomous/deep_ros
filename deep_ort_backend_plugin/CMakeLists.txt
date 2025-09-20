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
project(deep_ort_backend_plugin)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_link_options(-Wl,-no-undefined)
endif()

find_package(ament_cmake REQUIRED)
find_package(deep_core REQUIRED)
find_package(onnxruntime_vendor REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

# deep_ort_backend_plugin library
set(DEEP_ORT_LIB ${PROJECT_NAME}_lib)
add_library(${DEEP_ORT_LIB} SHARED
  src/ort_cpu_memory_allocator.cpp
  src/ort_backend_executor.cpp
  src/ort_backend_plugin.cpp
)

target_include_directories(${DEEP_ORT_LIB} PUBLIC
  $<BUILD_INTERFACE:${include_dir}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${DEEP_ORT_LIB}
  PUBLIC
    deep_core::deep_core_lib
    pluginlib::pluginlib
    rclcpp::rclcpp
  PRIVATE
    onnxruntime_vendor::onnxruntime_lib
)

install(TARGETS
  ${DEEP_ORT_LIB}
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

if(BUILD_TESTING)
  find_package(deep_test REQUIRED)

  add_deep_test(test_ort_backend test/test_ort_backend.cpp
    LIBRARIES ${DEEP_ORT_LIB}
  )

endif()

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_package()