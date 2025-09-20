# deep_core

Core package for the Deep ROS inference framework providing abstract interfaces, tensor operations, and plugin architecture for machine learning inference in ROS 2.

## Overview

`deep_core` provides the foundational components for a modular, high-performance ML inference system:

- **Tensor abstraction** with custom memory allocators
- **Plugin interfaces** for backend inference engines
- **Lifecycle node base class** for inference nodes
- **Type system** for tensor data types

## Architecture

### Core Components

- **`Tensor`**: Multi-dimensional array with pluggable memory allocators
- **`DeepNodeBase`**: ROS 2 lifecycle node base class for inference services
- **Plugin Interfaces**: Abstract base classes for backend implementations
  - `BackendMemoryAllocator`: Custom memory allocation strategies
  - `BackendInferenceExecutor`: ML framework inference execution
  - `DeepBackendPlugin`: Combined backend plugin interface

### Memory Management

The tensor system supports custom memory allocators for optimal performance:

```cpp
// Create tensor with custom allocator
auto allocator = get_custom_allocator();
deep_ros::Tensor input({1, 3, 224, 224}, deep_ros::DataType::FLOAT32, allocator);
```

### Plugin Architecture

Backend implementations are loaded dynamically using ROS 2 pluginlib:

```cpp
// Load backend plugin
if (!load_plugin("onnxruntime_cpu")) {
  RCLCPP_ERROR(get_logger(), "Failed to load backend plugin");
}

// Run inference
deep_ros::Tensor output = run_inference(input_tensor);
```

## Usage

### Creating an Inference Node

```cpp
#include <deep_core/deep_node_base.hpp>

class MyInferenceNode : public deep_ros::DeepNodeBase
{
public:
  MyInferenceNode(const rclcpp::NodeOptions & options)
  : DeepNodeBase("my_inference_node", options)
  {
  }

protected:
  CallbackReturn on_configure_impl(const rclcpp_lifecycle::State & state) override
  {
    // Custom configuration logic
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate_impl(const rclcpp_lifecycle::State & state) override
  {
    // Start inference services
    return CallbackReturn::SUCCESS;
  }
};
```

### Custom Memory Allocator

```cpp
class MyCustomAllocator : public deep_ros::BackendMemoryAllocator
{
public:
  void * allocate(size_t bytes) override {
    // Custom allocation strategy (e.g., GPU memory, aligned allocation)
    return my_custom_malloc(bytes);
  }

  void deallocate(void * ptr) override {
    my_custom_free(ptr);
  }

  // Implement other required methods...
};
```

## Package Structure

```
deep_core/
├── include/deep_core/
│   ├── deep_node_base.hpp          # Lifecycle node base class
│   ├── types/
│   │   ├── tensor.hpp              # Tensor class and data types
│   │   └── data_type.hpp           # Enum for tensor data types
│   └── plugin_interfaces/
│       ├── backend_memory_allocator.hpp    # Memory allocator interface
│       ├── backend_inference_executor.hpp  # Inference executor interface
│       └── deep_backend_plugin.hpp         # Combined plugin interface
├── src/
│   ├── deep_node_base.cpp          # Lifecycle node implementation
│   └── tensor.cpp                  # Tensor operations
└── CMakeLists.txt
```

## Dependencies

- **ROS 2**: rclcpp, rclcpp_lifecycle
- **pluginlib**: Dynamic plugin loading
- **Standard C++17**: Modern C++ features

## Supported Data Types

- `FLOAT32`: 32-bit floating point
- `INT32`: 32-bit signed integer
- `INT64`: 64-bit signed integer
- `UINT8`: 8-bit unsigned integer

## Backend Plugin Development

To create a new backend plugin:

1. Implement the three interfaces:
   - `BackendMemoryAllocator`
   - `BackendInferenceExecutor`
   - `DeepBackendPlugin`

2. Create a `plugins.xml` file:

```xml
<library path="my_backend_plugin_lib">
  <class name="my_backend" type="my_namespace::MyBackendPlugin" base_class_type="deep_ros::DeepBackendPlugin">
    <description>My custom ML backend</description>
  </class>
</library>
```

1. Export the plugin in your `package.xml`:

```xml
<export>
  <deep_ort_backend_plugin plugin="${prefix}/plugins.xml" />
</export>
```

## Examples

See the [`deep_ort_backend_plugin`](../deep_ort_backend_plugin/) package for a complete ONNX Runtime backend implementation.

## License

Licensed under the Apache License, Version 2.0.
