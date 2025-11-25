# Build Fixes Summary

This document summarizes all the changes and steps taken to fix the build issues for the Deep ROS workspace.

## Overview

The initial build failed due to several missing dependencies and configuration issues. This document outlines each problem encountered and the solutions implemented.

## Issues Fixed

### 1. Missing Build Tools

**Problem:** CMake couldn't find build tools (`make`, `gcc`, `g++`), causing errors like:
```
CMake Error: CMake was unable to find a build program corresponding to "Unix Makefiles".  
CMAKE_MAKE_PROGRAM is not set.
CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
```

**Solution:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

**What this installs:**
- `gcc` - GNU C compiler
- `g++` - GNU C++ compiler
- `make` - Build automation tool
- `libc6-dev` - C library development files
- Other essential build tools

---

### 2. CUDA Library Path Issue

**Problem:** The `deep_ort_gpu_backend_plugin` package failed to build because CMake couldn't find CUDA runtime libraries. The libraries were located at `/usr/local/cuda/targets/x86_64-linux/lib/` instead of the standard `/usr/local/cuda/lib64/` path.

**Solution:** Updated `/workspaces/deep_ros/deep_ort_gpu_backend_plugin/CMakeLists.txt` to search in the correct CUDA installation paths.

**Changes made:**
- Added `/usr/local/cuda/targets/x86_64-linux/lib` to library search paths
- Added `/usr/local/cuda-12.2/targets/x86_64-linux/lib` and other version-specific paths
- Added Strategy 5: Direct path fallback that checks common CUDA installation paths

**Key modifications:**
```cmake
find_library(CUDA_RUNTIME_LIBRARY
  NAMES cudart
  PATHS
    /usr/local/cuda/targets/x86_64-linux/lib
    /usr/local/cuda-12.2/targets/x86_64-linux/lib
    /usr/local/cuda-12/targets/x86_64-linux/lib
    # ... other paths
)

# Strategy 5: Direct path check as absolute last resort
if(NOT CUDA_RUNTIME_LINKED)
  set(POSSIBLE_CUDA_PATHS
    /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12
    /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudart.so.12
    # ... other paths
  )
  foreach(CUDA_PATH ${POSSIBLE_CUDA_PATHS})
    if(EXISTS ${CUDA_PATH})
      target_link_libraries(${DEEP_ORT_LIB} PRIVATE ${CUDA_PATH})
      set(CUDA_RUNTIME_LINKED TRUE)
      break()
    endif()
  endforeach()
endif()
```

---

### 3. CUDA Header Check Issue

**Problem:** The build required both CUDA headers and runtime library, but headers weren't always available in runtime-only installations.

**Solution:** Made the header check optional in `CMakeLists.txt`:

```cmake
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
```

---


### 4. Missing cuDNN Library

**Problem:** Runtime error when trying to use CUDA:
```
libcudnn.so.9: cannot open shared object file: No such file or directory
Failed to load library libonnxruntime_providers_cuda.so
```

**Solution:**
```bash
sudo apt-get install -y libcudnn9-cuda-12
```

**Note:** This installs cuDNN version 9 for CUDA 12, which is required by ONNX Runtime's CUDA execution provider.

---

### 6. Model Path Configuration Issue

**Problem:** The config file used a relative path `./yolov8m.onnx` which didn't resolve correctly when the node ran from a different working directory.

**Solution:** Updated `/workspaces/deep_ros/deep_object_detection/config/object_detection_params.yaml`:

```yaml
# Before:
model_path: "./yolov8m.onnx"

# After:
model_path: "/workspaces/deep_ros/yolov8m.onnx"
```

**Note:** After changing the config file, the package must be rebuilt so the updated config is installed to the `install/` directory.

---

### 7. Documentation Updates

**Problem:** No documentation about required system dependencies.

**Solution:** Added a "System Dependencies" section to `/workspaces/deep_ros/DEVELOPING.md` documenting:
- Required build tools (`build-essential`)
- Installation commands
- Verification steps
- Build instructions

---

## Files Modified

1. **`/workspaces/deep_ros/deep_ort_gpu_backend_plugin/CMakeLists.txt`**
   - Added CUDA library search paths
   - Added direct path fallback strategy
   - Made header check optional

2. **`/workspaces/deep_ros/deep_object_detection/src/ort_backend_inference.cpp`**
   - Enabled CUDA execution provider as primary option
   - Implemented fallback chain: CUDA → TensorRT → CPU

3. **`/workspaces/deep_ros/deep_object_detection/config/object_detection_params.yaml`**
   - Changed model path from relative to absolute

4. **`/workspaces/deep_ros/DEVELOPING.md`**
   - Added "System Dependencies" section

---

## Complete Build Sequence

Here's the complete sequence of commands to build the workspace from scratch:

```bash
# 1. Install build tools
sudo apt-get update
sudo apt-get install -y build-essential

# 2. Install cuDNN (for GPU support)
sudo apt-get install -y libcudnn9-cuda-12

# 3. Navigate to workspace and source ROS
cd /workspaces/deep_ros
source /opt/ros/humble/setup.bash

# 4. Build all packages
colcon build

# 5. Source workspace and run
source install/setup.bash
ros2 launch deep_object_detection simple_detection.launch.yaml
```



