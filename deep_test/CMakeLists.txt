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
project(deep_test)

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
find_package(deep_core REQUIRED)

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

# deep_test library
set(DEEP_TEST_LIB ${PROJECT_NAME}_lib)
add_library(${DEEP_TEST_LIB} SHARED
  src/test_executor_fixture.cpp
  src/mock_plugin/mock_plugin.cpp
)
target_include_directories(${DEEP_TEST_LIB} PUBLIC
  $<BUILD_INTERFACE:${include_dir}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${DEEP_TEST_LIB}
  PUBLIC
    deep_core::deep_core_lib
  PRIVATE
    rclcpp::rclcpp
    rclcpp_lifecycle::rclcpp_lifecycle
    pluginlib::pluginlib
)

install(TARGETS
  ${DEEP_TEST_LIB}
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

install(FILES cmake/add_deep_test.cmake
  DESTINATION share/${PROJECT_NAME}/cmake
)

# Install the mock plugin description file
install(FILES src/mock_plugin/plugins.xml
  DESTINATION share/${PROJECT_NAME}
)

# Export plugin description file to ament index for the mock plugin
pluginlib_export_plugin_description_file(deep_core src/mock_plugin/plugins.xml)

if(BUILD_TESTING)
  # Include our custom test function
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/add_deep_test.cmake)

  # Find test dependencies
  find_package(std_msgs REQUIRED)
  find_package(std_srvs REQUIRED)

  # Add the basic example test
  add_deep_test(test_basic_example
    test/test_basic_example.cpp
    LIBRARIES
      ${std_msgs_TARGETS}
      ${std_srvs_TARGETS}
  )

  # Add ROS test with multiple library dependencies
  add_deep_test(test_with_ros
    test/test_with_ros.cpp
    LIBRARIES
      ${std_msgs_TARGETS}
  )

  # Deep core tests - organized by functionality
  add_deep_test(test_tensor
    test/deep_core/test_tensor.cpp
    LIBRARIES
      ${DEEP_TEST_LIB}
  )
  add_deep_test(test_plugin_system
    test/deep_core/test_plugin_system.cpp
    LIBRARIES
      ${DEEP_TEST_LIB}
  )
  add_deep_test(test_node_lifecycle
    test/deep_core/test_node_lifecycle.cpp
    LIBRARIES
      ${DEEP_TEST_LIB}
  )
  add_deep_test(test_integration
    test/deep_core/test_integration.cpp
    LIBRARIES
      ${DEEP_TEST_LIB}
  )
endif()

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_package(
  CONFIG_EXTRAS "${PROJECT_NAME}-extras.cmake"
)
