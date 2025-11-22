# deep_ort_gpu_backend_plugin

ONNX Runtime GPU backend plugin for deep_core.

## Overview

Provides:
- GPU inference executor using ONNX Runtime with options for CUDA or TensorRT(untested) execution provider
- Device context management for multi-GPU systems

## Plugin Name

`onnxruntime_gpu`

## Supported Formats

ONNX models (.onnx files)

## Usage

Add to your `package.xml`:

```xml
<exec_depend>deep_ort_gpu_backend_plugin</exec_depend>
```

Configure your inference nodes to use this plugin:

```yaml
inference_node:
  ros__parameters:
    Backend.plugin: "onnxruntime_gpu"
    model_path: "/path/to/model.onnx"
```

## Dependencies

- deep_core
- onnxruntime_gpu_vendor
