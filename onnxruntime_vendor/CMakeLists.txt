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
cmake_minimum_required(VERSION 3.8)
project(onnxruntime_vendor VERSION 1.22.0)

find_package(ament_cmake REQUIRED)

set(ONNXRUNTIME_VERSION "${PROJECT_VERSION}")
set(ONNXRUNTIME_URL
  "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
)
set(ONNXRUNTIME_DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-download")
set(ONNXRUNTIME_EXTRACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")

# Download and extract ONNX Runtime
if(NOT EXISTS "${ONNXRUNTIME_EXTRACT_DIR}")
  message(STATUS "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION}...")
  file(DOWNLOAD
    ${ONNXRUNTIME_URL}
    "${ONNXRUNTIME_DOWNLOAD_DIR}/onnxruntime.tgz"
    SHOW_PROGRESS
    STATUS DOWNLOAD_STATUS
  )

  list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
  if(NOT DOWNLOAD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to download ONNX Runtime")
  endif()

  message(STATUS "Extracting ONNX Runtime...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${ONNXRUNTIME_DOWNLOAD_DIR}/onnxruntime.tgz"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE EXTRACT_RESULT
  )

  if(NOT EXTRACT_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to extract ONNX Runtime")
  endif()
endif()

# Install headers
install(
  DIRECTORY "${ONNXRUNTIME_EXTRACT_DIR}/include/"
  DESTINATION include
)

# Install libraries
install(
  DIRECTORY "${ONNXRUNTIME_EXTRACT_DIR}/lib/"
  DESTINATION lib
  PATTERN "*.a" EXCLUDE
)

# Create an interface library for users to link against
add_library(onnxruntime_lib INTERFACE)
target_include_directories(onnxruntime_lib INTERFACE
  $<BUILD_INTERFACE:${ONNXRUNTIME_EXTRACT_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(onnxruntime_lib INTERFACE
  $<BUILD_INTERFACE:${ONNXRUNTIME_EXTRACT_DIR}/lib/libonnxruntime.so>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/lib/libonnxruntime.so>
)

# Export the target
install(
  TARGETS onnxruntime_lib
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

ament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies()

ament_package()
