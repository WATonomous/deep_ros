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

#
# Catch2 Unittest Function
# Creates a test executable with Catch2 support
#
# Usage:
#   add_deep_test(test_name test_source.cpp
#     LIBRARIES lib1 lib2 lib3
#   )
#
function(add_deep_test TEST_NAME TEST_SOURCE)
  # Parse arguments
  cmake_parse_arguments(
    PARSE_ARGV 2  # Start parsing from the 3rd argument
    ARG           # Prefix for parsed variables
    ""            # Options (flags with no values)
    ""            # One-value keywords
    "LIBRARIES"   # Multi-value keywords
  )

  # Find dependencies
  # Catch2 version requirement compatibility:
  # Humble uses Catch2 2.x, Jazzy+ uses Catch2 3.x
  if("$ENV{ROS_DISTRO}" STREQUAL "humble")
    find_package(Catch2 2 REQUIRED)
  else()
    find_package(Catch2 REQUIRED)
  endif()
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_lifecycle REQUIRED)

  # Create the test executable
  add_executable(${TEST_NAME} ${TEST_SOURCE})

  # Determine deep_test library target name
  # When building within deep_test package: use local target
  # When building in external packages: use exported target
  if(TARGET deep_test_lib)
    set(DEEP_TEST_LIB_TARGET deep_test_lib)
  else()
    set(DEEP_TEST_LIB_TARGET deep_test::deep_test_lib)
  endif()

  # Link with Catch2, ROS dependencies, deep_test library, and specified libraries
  target_link_libraries(${TEST_NAME}
    Catch2::Catch2WithMain
    rclcpp::rclcpp
    rclcpp_lifecycle::rclcpp_lifecycle
    ${DEEP_TEST_LIB_TARGET}
    ${ARG_LIBRARIES}
  )

  # Add test include directory
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)

  # Register the test with ament
  ament_add_test(
    ${TEST_NAME}
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    COMMAND "$<TARGET_FILE:${TEST_NAME}>"
      -r junit
      -o test_results/${PROJECT_NAME}/${TEST_NAME}_output.xml
    ENV CATCH_CONFIG_CONSOLE_WIDTH=120
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
endfunction()


#
# Launch Test Function with test_utils support
# Wraps add_launch_test to automatically add test/launch_tests to PYTHONPATH
#
# Usage:
#   add_deep_launch_test(test/launch_tests/test_example.py)
#
macro(add_deep_launch_test filename)
  add_launch_test(
    ${filename}
    ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/test/launch_tests:$ENV{PYTHONPATH}
    ${ARGN}
  )
endmacro()
