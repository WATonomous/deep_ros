# deep_ort_backend_plugin

ONNX Runtime backend plugin for the Deep ROS inference framework, providing high-performance CPU inference with zero-copy memory management and IO binding.

## Overview

This package implements a complete backend plugin for ONNX Runtime that integrates seamlessly with the `deep_core` framework. It provides:

- **Zero-copy inference** using ONNX Runtime IO binding
- **Custom memory allocator** with 64-byte alignment for SIMD optimization
- **CPU-optimized execution** with configurable thread pools
- **Dynamic shape handling** for variable batch sizes and input dimensions

## Features

### 🚀 Zero-Copy Performance
- Direct memory binding between `deep_ros::Tensor` and ONNX Runtime
- No memory copying during inference execution
- Optimal memory layout with 64-byte alignment

### 🧠 Smart Shape Inference
- Automatic output shape detection from ONNX model metadata
- Dynamic dimension resolution using input shapes
- Support for variable batch sizes and flexible input sizes

### ⚡ Optimized Execution
- CPU execution provider with extended optimizations
- Configurable intra-op thread pools
- Memory-efficient arena allocators

## Architecture

### Components

1. **`OrtCpuMemoryAllocator`**: 64-byte aligned memory allocator
2. **`OrtBackendExecutor`**: ONNX Runtime inference execution engine
3. **`OrtBackendPlugin`**: Combined plugin interface implementation

### Memory Flow

```
Input: deep_ros::Tensor (custom allocator)
    ↓ (zero-copy wrap)
ONNX Runtime: Ort::Value 
    ↓ (IO binding)
Inference: session_->Run(binding)
    ↓ (zero-copy wrap)
Output: deep_ros::Tensor (custom allocator)
```

## Usage

### Loading the Plugin

```cpp
#include <deep_core/deep_node_base.hpp>

class MyInferenceNode : public deep_ros::DeepNodeBase
{
public:
  MyInferenceNode() : DeepNodeBase("inference_node") {
    // Load ONNX Runtime backend
    if (!load_plugin("onnxruntime_cpu")) {
      RCLCPP_ERROR(get_logger(), "Failed to load ONNX Runtime backend");
    }
    
    // Load model
    if (!load_model("/path/to/model.onnx")) {
      RCLCPP_ERROR(get_logger(), "Failed to load ONNX model");
    }
  }
  
  void process_input(const sensor_msgs::msg::Image& image) {
    // Convert image to tensor
    auto input = image_to_tensor(image);
    
    // Run inference (zero-copy!)
    auto output = run_inference(input);
    
    // Process results
    publish_results(output);
  }
};
```

### Direct Plugin Usage

```cpp
#include <deep_ort_backend_plugin/ort_backend_plugin.hpp>

// Create plugin instance
auto plugin = std::make_unique<deep_ort_backend::OrtBackendPlugin>();

// Get allocator and executor
auto allocator = plugin->get_allocator();
auto executor = plugin->get_inference_executor();

// Load model
executor->load_model("/path/to/model.onnx");

// Create input tensor with custom allocator
deep_ros::Tensor input({1, 3, 224, 224}, deep_ros::DataType::FLOAT32, allocator);

// Fill input data...
fill_input_data(input);

// Run inference
auto output = executor->run_inference(input);
```

## Configuration

### ROS Parameters

Configure the backend through ROS parameters:

```yaml
inference_node:
  ros__parameters:
    backend: "onnxruntime_cpu"
    model_path: "/path/to/your/model.onnx"
```

### Launch File Example

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='your_package',
            executable='inference_node',
            name='object_detector',
            parameters=[{
                'backend': 'onnxruntime_cpu',
                'model_path': '/models/yolo_v8n.onnx'
            }]
        )
    ])
```

## Performance Features

### Memory Alignment
- **64-byte alignment**: Optimized for AVX-512 SIMD instructions
- **Zero fragmentation**: Efficient memory pool management
- **Cache-friendly**: Optimal memory access patterns

### ONNX Runtime Optimizations
- **Graph optimization**: Extended optimization level enabled
- **Thread pool**: Configurable intra-op parallelism
- **Arena allocator**: Memory-efficient allocation strategy

### Dynamic Shape Support

```cpp
// Works with any input shape
deep_ros::Tensor input1({1, 3, 224, 224}, dtype, allocator);    // Batch 1
deep_ros::Tensor input2({4, 3, 224, 224}, dtype, allocator);    // Batch 4
deep_ros::Tensor input3({1, 3, 512, 512}, dtype, allocator);    // Different resolution

// Output shapes are automatically inferred
auto output1 = executor->run_inference(input1);  // [1, 1000]
auto output2 = executor->run_inference(input2);  // [4, 1000] 
auto output3 = executor->run_inference(input3);  // [1, 1000]
```

## Model Compatibility

### Supported Formats
- **ONNX models** (.onnx files)
- **All ONNX opsets** supported by ONNX Runtime 1.22.0
- **Dynamic shapes** with `-1` dimensions

### Tested Model Types
- ✅ **Image Classification** (ResNet, EfficientNet, Vision Transformers)
- ✅ **Object Detection** (YOLO, R-CNN, SSD)
- ✅ **Semantic Segmentation** (U-Net, DeepLab)
- ✅ **NLP Models** (BERT, GPT, T5)

## Package Structure

```
deep_ort_backend_plugin/
├── include/deep_ort_backend_plugin/
│   ├── ort_backend_plugin.hpp           # Main plugin interface
│   ├── ort_cpu_memory_allocator.hpp     # Custom memory allocator
│   └── ort_backend_executor.hpp         # Inference executor
├── src/
│   ├── ort_backend_plugin.cpp           # Plugin implementation
│   ├── ort_cpu_memory_allocator.cpp     # Allocator implementation
│   └── ort_backend_executor.cpp         # Executor with IO binding
├── plugins.xml                          # Plugin registration
└── CMakeLists.txt
```

## Dependencies

- **deep_core**: Core interfaces and tensor system
- **onnxruntime_vendor**: ONNX Runtime 1.22.0
- **pluginlib**: ROS 2 plugin system
- **rclcpp**: ROS 2 C++ client library

## Performance Benchmarks

### Memory Efficiency
- **Zero-copy overhead**: 0% (direct memory binding)
- **Memory alignment**: 64-byte for optimal SIMD performance  
- **Allocation strategy**: Arena-based for reduced fragmentation

### Inference Speed
- **CPU optimization**: Extended graph optimizations enabled
- **Thread utilization**: Configurable intra-op thread pools
- **Memory access**: Cache-optimized aligned allocations

## Troubleshooting

### Common Issues

1. **Plugin not found**:
   ```bash
   # Verify plugin registration
   ros2 pkg prefix deep_ort_backend_plugin
   ```

2. **Model loading fails**:
   ```bash
   # Check file permissions and path
   ls -la /path/to/model.onnx
   ```

3. **Memory alignment errors**:
   ```cpp
   // Ensure tensor size >= 128 bytes (ONNX Runtime requirement)
   if (tensor.size() * tensor.element_size() < 128) {
       // Use smaller data types or add padding
   }
   ```

## Future Enhancements

- [ ] **GPU backend**: CUDA and ROCm execution providers
- [ ] **Quantization**: INT8 and FP16 optimizations  
- [ ] **Model optimization**: Runtime graph optimization
- [ ] **Batch processing**: Efficient multi-request batching
- [ ] **Memory pools**: Advanced memory management strategies

## License

Licensed under the Apache License, Version 2.0.