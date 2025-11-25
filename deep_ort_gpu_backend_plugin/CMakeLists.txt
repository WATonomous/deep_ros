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
project(deep_ort_gpu_backend_plugin)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_link_options(-Wl,-no-undefined)
endif()

find_package(ament_cmake REQUIRED)
find_package(deep_core REQUIRED)

# Create dummy CUDA targets early to prevent onnxruntime from expecting real ones
if(NOT TARGET CUDA::cudart)
  add_library(CUDA::cudart INTERFACE IMPORTED)
  message(STATUS "Created dummy CUDA::cudart target")
endif()
if(NOT TARGET CUDA::cublas)
  add_library(CUDA::cublas INTERFACE IMPORTED)
  message(STATUS "Created dummy CUDA::cublas target")
endif()
if(NOT TARGET CUDA::curand)
  add_library(CUDA::curand INTERFACE IMPORTED)
  message(STATUS "Created dummy CUDA::curand target")
endif()
if(NOT TARGET CUDA::cufft)
  add_library(CUDA::cufft INTERFACE IMPORTED)
  message(STATUS "Created dummy CUDA::cufft target")
endif()
if(NOT TARGET CUDA::cusparse)
  add_library(CUDA::cusparse INTERFACE IMPORTED)
  message(STATUS "Created dummy CUDA::cusparse target")
endif()
if(NOT TARGET CUDA::cusolver)
  add_library(CUDA::cusolver INTERFACE IMPORTED)
  message(STATUS "Created dummy CUDA::cusolver target")
endif()

find_package(onnxruntime_gpu_vendor REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# Use CMake's built-in CUDA support with minimal requirements
find_package(CUDAToolkit QUIET)

# Check for specific CUDA libraries that might be required
if(CUDAToolkit_FOUND)
  # Check if all required CUDA components are available
  set(CUDA_LIBS_AVAILABLE TRUE)

  if(NOT TARGET CUDA::cudart)
    message(WARNING "CUDA::cudart not found")
    set(CUDA_LIBS_AVAILABLE FALSE)
  endif()

  # Some dependencies might require these, so check if they exist
  if(NOT TARGET CUDA::cublas)
    message(WARNING "CUDA::cublas not found - will try to continue without it")
  endif()

  if(NOT TARGET CUDA::curand)
    message(WARNING "CUDA::curand not found - will try to continue without it")
  endif()

  if(CUDA_LIBS_AVAILABLE)
    message(STATUS "Found CUDA using CUDAToolkit with cudart")
    set(CUDA_FOUND TRUE)
  else()
    message(WARNING "CUDAToolkit found but missing essential components")
    set(CUDAToolkit_FOUND FALSE)
    set(CUDA_FOUND FALSE)
  endif()
endif()

# Fallback: Try to find CUDA manually if CUDAToolkit fails or is incomplete
if(NOT CUDAToolkit_FOUND)
  # Find CUDA runtime libraries (without requiring nvcc/development toolkit)
  # Try multiple possible locations for CUDA headers and libraries
  find_path(CUDA_INCLUDE_DIR
    NAMES cuda_runtime.h
    PATHS
      /local/cuda/include
      /local/cuda-12.6/include
      /usr/local/cuda-12.6/targets/x86_64-linux/include
      /usr/local/cuda/targets/x86_64-linux/include
      /usr/include
      /usr/include/cuda
      /usr/local/cuda/include
      /usr/local/cuda-12/include
      /usr/local/cuda-12.6/include
      /opt/cuda/include
    DOC "CUDA include directory"
  )

  find_library(CUDA_RUNTIME_LIBRARY
    NAMES cudart
    PATHS
      /local/cuda/lib64
      /local/cuda-12.6/lib64
      /usr/lib/x86_64-linux-gnu
      /usr/local/cuda/targets/x86_64-linux/lib
      /usr/local/cuda-12.2/targets/x86_64-linux/lib
      /usr/local/cuda-12/targets/x86_64-linux/lib
      /usr/local/cuda-12.6/targets/x86_64-linux/lib
      /usr/local/cuda/lib64
      /usr/local/cuda-12/lib64
      /usr/local/cuda-12.6/lib64
      /usr/local/cuda/lib
      /opt/cuda/lib64
    DOC "CUDA runtime library"
  )

  if(CUDA_RUNTIME_LIBRARY)
    message(STATUS "Found CUDA runtime: ${CUDA_RUNTIME_LIBRARY}")
    if(CUDA_INCLUDE_DIR)
      message(STATUS "Found CUDA headers: ${CUDA_INCLUDE_DIR}")
    else()
      message(WARNING "CUDA headers not found, but runtime library found - will use runtime library only")
    endif()
    set(CUDA_FOUND TRUE)
  else()
    message(WARNING "CUDA runtime not found - skipping tests that require CUDA")
    set(CUDA_FOUND FALSE)
  endif()
else()
  message(STATUS "Found CUDA using CUDAToolkit")
  set(CUDA_FOUND TRUE)
endif()

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

# deep_ort_gpu_backend_plugin library
set(DEEP_ORT_LIB ${PROJECT_NAME}_lib)
add_library(${DEEP_ORT_LIB} SHARED
  src/ort_gpu_memory_allocator.cpp
  src/ort_gpu_backend_executor.cpp
  src/ort_gpu_backend_plugin.cpp
)

target_include_directories(${DEEP_ORT_LIB} PUBLIC
  $<BUILD_INTERFACE:${include_dir}>
  $<INSTALL_INTERFACE:include>
)

# Add common link libraries (always present)
target_link_libraries(${DEEP_ORT_LIB}
  PUBLIC
    pluginlib::pluginlib
    rclcpp::rclcpp
  PRIVATE
    deep_core::deep_core_lib
    onnxruntime_gpu_vendor::onnxruntime_gpu_lib
)


# CUDA runtime is required since code calls CUDA functions directly
# Try multiple strategies to find and link CUDA runtime
set(CUDA_RUNTIME_LINKED FALSE)

# Strategy 1: Use manually found CUDA runtime
if(CUDA_FOUND AND CUDA_RUNTIME_LIBRARY AND NOT CUDAToolkit_FOUND)
  target_link_libraries(${DEEP_ORT_LIB} PRIVATE ${CUDA_RUNTIME_LIBRARY})
  message(STATUS "Linking manual CUDA runtime: ${CUDA_RUNTIME_LIBRARY}")
  set(CUDA_RUNTIME_LINKED TRUE)
endif()

# Strategy 2: Use CUDAToolkit's cudart target
if(NOT CUDA_RUNTIME_LINKED AND CUDAToolkit_FOUND)
  # Override the dummy target with the real one if available
  if(TARGET CUDA::cudart_static)
    target_link_libraries(${DEEP_ORT_LIB} PRIVATE CUDA::cudart_static)
    message(STATUS "Linking CUDAToolkit cudart_static")
    set(CUDA_RUNTIME_LINKED TRUE)
  elseif(TARGET CUDA::cudart)
    # Check if it's not just our dummy target
    get_target_property(CUDART_TYPE CUDA::cudart TYPE)
    if(NOT CUDART_TYPE STREQUAL "INTERFACE_LIBRARY")
      target_link_libraries(${DEEP_ORT_LIB} PRIVATE CUDA::cudart)
      message(STATUS "Linking CUDAToolkit cudart")
      set(CUDA_RUNTIME_LINKED TRUE)
    endif()
  endif()
endif()

# Strategy 3: Search more aggressively for any CUDA runtime library
if(NOT CUDA_RUNTIME_LINKED)
  find_library(AGGRESSIVE_CUDA_RUNTIME_LIBRARY
    NAMES cudart libcudart cudart.so cudart.so.12 cudart.so.11 cudart.so.10
    PATHS
      /local/cuda/lib64
      /local/cuda-12.6/lib64
      /local/cuda-12/lib64
      /local/cuda-11/lib64
      /usr/lib/x86_64-linux-gnu
      /usr/local/cuda/targets/x86_64-linux/lib
      /usr/local/cuda-12.2/targets/x86_64-linux/lib
      /usr/local/cuda-12/targets/x86_64-linux/lib
      /usr/local/cuda-12.6/targets/x86_64-linux/lib
      /usr/local/cuda/lib64
      /usr/local/cuda-12/lib64
      /usr/local/cuda-12.6/lib64
      /usr/local/cuda-11/lib64
      /usr/local/cuda/lib
      /opt/cuda/lib64
      /usr/lib64
      /lib64
      /lib/x86_64-linux-gnu
    NO_DEFAULT_PATH
  )

  if(AGGRESSIVE_CUDA_RUNTIME_LIBRARY)
    target_link_libraries(${DEEP_ORT_LIB} PRIVATE ${AGGRESSIVE_CUDA_RUNTIME_LIBRARY})
    message(STATUS "Linking aggressive search CUDA runtime: ${AGGRESSIVE_CUDA_RUNTIME_LIBRARY}")
    set(CUDA_RUNTIME_LINKED TRUE)
  endif()
endif()

# Strategy 4: Try system-wide search as last resort
if(NOT CUDA_RUNTIME_LINKED)
  find_library(SYSTEM_CUDA_RUNTIME_LIBRARY NAMES cudart)
  if(SYSTEM_CUDA_RUNTIME_LIBRARY)
    target_link_libraries(${DEEP_ORT_LIB} PRIVATE ${SYSTEM_CUDA_RUNTIME_LIBRARY})
    message(STATUS "Linking system CUDA runtime: ${SYSTEM_CUDA_RUNTIME_LIBRARY}")
    set(CUDA_RUNTIME_LINKED TRUE)
  endif()
endif()

# Strategy 5: Direct path check as absolute last resort
if(NOT CUDA_RUNTIME_LINKED)
  # Check common CUDA installation paths directly
  set(POSSIBLE_CUDA_PATHS
    /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12
    /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudart.so.12
    /usr/local/cuda-12/targets/x86_64-linux/lib/libcudart.so.12
    /usr/local/cuda/lib64/libcudart.so.12
    /usr/local/cuda/lib/libcudart.so.12
  )
  
  foreach(CUDA_PATH ${POSSIBLE_CUDA_PATHS})
    if(EXISTS ${CUDA_PATH})
      target_link_libraries(${DEEP_ORT_LIB} PRIVATE ${CUDA_PATH})
      message(STATUS "Linking CUDA runtime from direct path: ${CUDA_PATH}")
      set(CUDA_RUNTIME_LINKED TRUE)
      break()
    endif()
  endforeach()
endif()

if(NOT CUDA_RUNTIME_LINKED)
  message(FATAL_ERROR "No CUDA runtime found - required for GPU functionality")
endif()

# Add CUDA include directories if found
if(CUDAToolkit_FOUND)
  target_include_directories(${DEEP_ORT_LIB} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
elseif(CUDA_FOUND)
  target_include_directories(${DEEP_ORT_LIB} PRIVATE ${CUDA_INCLUDE_DIR})
endif()

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

# Export plugin description file to ament index
pluginlib_export_plugin_description_file(deep_core plugins.xml)

# if(BUILD_TESTING)
#   find_package(deep_test REQUIRED)

#   # add_deep_test(test_ort_gpu_backend test/test_ort_gpu_backend.cpp
#   #   LIBRARIES
#   #     ${DEEP_ORT_LIB}
#   #     deep_core::deep_core_lib
#   #     onnxruntime_gpu_vendor::onnxruntime_gpu_lib
#   # )

# endif()

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_libraries(${DEEP_ORT_LIB})
ament_package()
