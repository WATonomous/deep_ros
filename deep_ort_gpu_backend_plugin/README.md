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

## Current problems

  1. No proper IO binding - Despite documentation claiming "zero-copy," the code doesn't use Ort::IoBinding. The CPU backend does this correctly but GPU backend doesn't.
  2. Thread-local caching bug (ort_gpu_backend_executor.cpp:104-108) - Input/output names are cached as static thread_local, which will break if models are reloaded or multiple instances exist.
  3. Hardcoded float types (ort_gpu_backend_executor.cpp:114,139) - Input/output tensors are hardcoded to <float>, ignoring the actual data type, which will fail for non-float models.
  4. Stub implementations - Methods like verify_gpu_availability() and set_device() are empty/always return true.
  5. Unused member - persistent_cuda_ptr_ is declared but never used, suggesting incomplete GPU memory management.
