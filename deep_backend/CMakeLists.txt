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
project(deep_backend)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_link_options(-Wl,-no-undefined)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(deep_tensor REQUIRED)

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

# deep_backend library
set(DEEP_TENSOR_LIB ${PROJECT_NAME}_lib)

target_include_directories(${DEEP_TENSOR_LIB} PUBLIC
  $<BUILD_INTERFACE:${include_dir}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${DEEP_TENSOR_LIB}
  PUBLIC
    rclcpp::rclcpp
    deep_tensor::deep_tensor
)

install(TARGETS
  ${DEEP_TENSOR_LIB}
  EXPORT ${DEEP_TENSOR_LIB}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(EXPORT ${DEEP_TENSOR_LIB}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)

install(DIRECTORY include/
  DESTINATION include
)

# if(BUILD_TESTING)
#   find_package(deep_test REQUIRED)

# endif()

ament_export_targets(${DEEP_TENSOR_LIB}Targets HAS_LIBRARY_TARGET)
ament_package()
