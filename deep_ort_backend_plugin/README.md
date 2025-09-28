# deep_ort_backend_plugin

ONNX Runtime CPU backend plugin for deep_core.

## Overview

Provides:
- CPU inference executor using ONNX Runtime
- 64-byte aligned memory allocator for optimal SIMD performance
- Zero-copy inference with IO binding

## Plugin Name

`onnxruntime_cpu`

## Supported Formats

ONNX models (.onnx files)

## Usage

Add to your `package.xml`:

```xml
<exec_depend>deep_ort_backend_plugin</exec_depend>
```

Configure your inference nodes to use this plugin:

```yaml
inference_node:
  ros__parameters:
    Backend.plugin: "onnxruntime_cpu"
    model_path: "/path/to/model.onnx"
```

## Dependencies

- deep_core
- onnxruntime_vendor
