# Copyright (c) 2025-present Polymath Robotics, Inc. All rights reserved
# Proprietary. Any unauthorized copying, distribution, or modification of this software is strictly prohibited.
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

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

# deep_test library
set(DEEP_TEST_LIB ${PROJECT_NAME}_lib)
add_library(${DEEP_TEST_LIB} SHARED
  src/test_executor_fixture.cpp
)
target_include_directories(${DEEP_TEST_LIB} PUBLIC
  $<BUILD_INTERFACE:${include_dir}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${DEEP_TEST_LIB}
  PRIVATE
    rclcpp::rclcpp
    rclcpp_lifecycle::rclcpp_lifecycle
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
      ${DEEP_TEST_LIB}
      ${std_msgs_TARGETS}
      ${std_srvs_TARGETS}
  )
  
  # Add ROS test with multiple library dependencies
  add_deep_test(test_with_ros
    test/test_with_ros.cpp
    LIBRARIES 
      ${DEEP_TEST_LIB}
      ${std_msgs_TARGETS}
  )
endif()

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_package(
  CONFIG_EXTRAS "${PROJECT_NAME}-extras.cmake"
)
