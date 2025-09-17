cmake_minimum_required(VERSION 3.8)
project(deep_object_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# Try to find CUDA and TensorRT (optional for development)
# Find CUDA with proper paths
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.2/bin/nvcc)
find_package(CUDA REQUIRED)

# Add CUDA include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda-12.2/include)
find_package(PkgConfig QUIET)

set(TENSORRT_FOUND FALSE)
if(CUDA_FOUND)
  pkg_check_modules(TENSORRT nvinfer)
  if(TENSORRT_FOUND)
    set(USE_TENSORRT TRUE)
    add_definitions(-DUSE_TENSORRT)
  else()
    message(WARNING "TensorRT not found. Building without GPU acceleration.")
    set(USE_TENSORRT FALSE)
  endif()
else()
  message(WARNING "CUDA not found. Building without GPU acceleration.")
  set(USE_TENSORRT FALSE)
endif()

# Include directories
include_directories(include)

if(USE_TENSORRT)
  include_directories(${CUDA_INCLUDE_DIRS})
  include_directories(${TENSORRT_INCLUDE_DIRS})
endif()

# Add executable
add_executable(object_detection_node
  src/object_detection_node.cpp
  src/tensorrt_inference.cpp
)

# Target dependencies
ament_target_dependencies(object_detection_node
  rclcpp
  sensor_msgs
  vision_msgs
  visualization_msgs
  std_msgs
  cv_bridge
  image_transport
  message_filters
  OpenCV
)

# Link libraries
target_link_libraries(object_detection_node yaml-cpp)

if(USE_TENSORRT)
  target_link_libraries(object_detection_node
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARIES}
    nvinfer
    nvonnxparser
    cudart
  )
endif()

# Install targets
install(TARGETS
  object_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install directories
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()