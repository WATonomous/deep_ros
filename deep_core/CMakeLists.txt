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
project(deep_core)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_link_options(-Wl,-no-undefined)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
find_package(bondcpp REQUIRED)

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

# deep_core library
set(DEEP_CORE_LIB ${PROJECT_NAME}_lib)
add_library(${DEEP_CORE_LIB} SHARED
  src/deep_node_base.cpp
  src/tensor.cpp
)
target_include_directories(${DEEP_CORE_LIB} PUBLIC
  $<BUILD_INTERFACE:${include_dir}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${DEEP_CORE_LIB}
  PUBLIC
    rclcpp::rclcpp
    rclcpp_lifecycle::rclcpp_lifecycle
    pluginlib::pluginlib
    bondcpp::bondcpp
)

install(TARGETS
  ${DEEP_CORE_LIB}
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

if(BUILD_TESTING)
  find_package(deep_test REQUIRED)

  add_deep_test(test_deep_core test/test_deep_core.cpp
    LIBRARIES
      ${DEEP_CORE_LIB}
  )

endif()

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_package()
