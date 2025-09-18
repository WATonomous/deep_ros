# onnxruntime_gpu_vendor

ROS2 vendor package for ONNX Runtime (GPU/CUDA/TensorRT)

## Usage

### Package.xml

```xml
<depend>onnxruntime_gpu_vendor</depend>
```

### CMakeLists.txt

```cmake
find_package(onnxruntime_gpu_vendor REQUIRED)

add_executable(my_node src/my_node.cpp)
target_link_libraries(my_node
  onnxruntime_gpu_vendor::onnxruntime_gpu_lib
)
```

### C++ Code

```cpp
#include <onnxruntime_cxx_api.h>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "my_app");
Ort::SessionOptions session_options;

// Optional: Enable CUDA
OrtCUDAProviderOptions cuda_options;
cuda_options.device_id = 0;
session_options.AppendExecutionProvider_CUDA(cuda_options);
```
