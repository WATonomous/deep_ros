# onnxruntime_vendor

ROS2 vendor package for ONNX Runtime (CPU)

## Usage

### Package.xml

```xml
<depend>onnxruntime_vendor</depend>
```

### CMakeLists.txt

```cmake
find_package(onnxruntime_vendor REQUIRED)

add_executable(my_node src/my_node.cpp)
target_link_libraries(my_node
  onnxruntime_vendor::onnxruntime_lib
)
```

### C++ Code

```cpp
#include <onnxruntime_cxx_api.h>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "my_app");
Ort::SessionOptions session_options;
// Load model, run inference...
```
