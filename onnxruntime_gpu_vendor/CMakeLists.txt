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
project(onnxruntime_gpu_vendor VERSION 1.22.0)

find_package(ament_cmake REQUIRED)

# Find CUDA libraries (optional, don't fail if not found)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
else()
  message(WARNING "CUDA not found - GPU acceleration may not work")
endif()

set(ONNXRUNTIME_VERSION "${PROJECT_VERSION}")

# Package flavor: "gpu" (CUDA-only) or "tensorrt" (CUDA + TensorRT EP, if available for this version).
set(ONNXRUNTIME_FLAVOR "gpu" CACHE STRING "ONNX Runtime package flavor (gpu|tensorrt)")
# Recent ONNX Runtime releases no longer publish a separate TensorRT archive.
# The standard GPU archive already contains the TensorRT provider library, so fall back to it.
set(ONNXRUNTIME_PACKAGE_FLAVOR "${ONNXRUNTIME_FLAVOR}")
if(ONNXRUNTIME_FLAVOR STREQUAL "tensorrt")
  set(ONNXRUNTIME_PACKAGE_FLAVOR "gpu")
  message(WARNING
    "TensorRT flavor requested; using the GPU archive instead because upstream no longer ships a separate TensorRT package. "
    "Ensure TensorRT runtime libraries are installed on the system.")
endif()
set(ONNXRUNTIME_URL_DEFAULT
  "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_PACKAGE_FLAVOR}-${ONNXRUNTIME_VERSION}.tgz"
)
set(ONNXRUNTIME_URL "${ONNXRUNTIME_URL_DEFAULT}" CACHE STRING "URL to prebuilt ONNX Runtime archive")
set(ONNXRUNTIME_DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-download")
set(ONNXRUNTIME_EXTRACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-linux-x64-${ONNXRUNTIME_PACKAGE_FLAVOR}-${ONNXRUNTIME_VERSION}")

# Download and extract ONNX Runtime GPU
if(NOT EXISTS "${ONNXRUNTIME_EXTRACT_DIR}")
  message(STATUS "Downloading ONNX Runtime GPU ${ONNXRUNTIME_VERSION}...")
  file(MAKE_DIRECTORY "${ONNXRUNTIME_DOWNLOAD_DIR}")
  file(DOWNLOAD
    ${ONNXRUNTIME_URL}
    "${ONNXRUNTIME_DOWNLOAD_DIR}/onnxruntime-gpu.tgz"
    SHOW_PROGRESS
    STATUS DOWNLOAD_STATUS
  )

  list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
  list(GET DOWNLOAD_STATUS 1 DOWNLOAD_MESSAGE)
  if(NOT DOWNLOAD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to download ONNX Runtime from ${ONNXRUNTIME_URL}: ${DOWNLOAD_MESSAGE}")
  endif()

  message(STATUS "Extracting ONNX Runtime GPU...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${ONNXRUNTIME_DOWNLOAD_DIR}/onnxruntime-gpu.tgz"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE EXTRACT_RESULT
  )

  if(NOT EXTRACT_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to extract ONNX Runtime GPU")
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

# Export library paths for runtime
ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/env_hook/onnxruntime_library_path.sh.in"
)

# Create an interface library for users to link against
add_library(onnxruntime_gpu_lib INTERFACE)
target_include_directories(onnxruntime_gpu_lib INTERFACE
  $<BUILD_INTERFACE:${ONNXRUNTIME_EXTRACT_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(onnxruntime_gpu_lib INTERFACE
  $<BUILD_INTERFACE:${ONNXRUNTIME_EXTRACT_DIR}/lib/libonnxruntime.so>
  $<BUILD_INTERFACE:${ONNXRUNTIME_EXTRACT_DIR}/lib/libonnxruntime_providers_shared.so>
  $<BUILD_INTERFACE:${ONNXRUNTIME_EXTRACT_DIR}/lib/libonnxruntime_providers_cuda.so>
  $<BUILD_INTERFACE:${ONNXRUNTIME_EXTRACT_DIR}/lib/libonnxruntime_providers_tensorrt.so>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/lib/libonnxruntime.so>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/lib/libonnxruntime_providers_shared.so>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/lib/libonnxruntime_providers_cuda.so>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/lib/libonnxruntime_providers_tensorrt.so>
)

# Export the target
install(
  TARGETS onnxruntime_gpu_lib
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

ament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies()

ament_package()
