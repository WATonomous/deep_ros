# Deep ROS

Full ML infrastructure pipeline for ROS2. Includes inference model-agnostic node containers for quick deployment and testing of ML models, as well as sample model farm for building, training, evaluating, and quantizing neural networks.

## Installation

(TODO) Add the base library into the ROS buildfarm

```bash
sudo apt install ros-${ROS_DISTRO}-deep-ros
```

## Backend Plugin Installation
To accomodate different hardware accelerators, deep ros has a library of installable plugins that deal with model loading, memory allocation, and inference for a specific hardware accelerator.

To configure the backend your node should run with, specify the backend in the node's parameters:

```yaml
sample_inference_node:
  ros__parameters:
    # Backend configuration - TensorRT
    Backend:
      plugin: "onnxruntime_gpu"
      device_id: 0
      execution_provider: "tensorrt"
```

Each backend has its own subset of parameters.

### `deep_ort_backend_plugin`
ONNXRuntime CPU Backend. This comes with the base library.

In your package.xml

```xml
<exec_depend>deep_ort_backend_plugin</exec_depend>
```

Specify in your parameter file

```yaml
```yaml
sample_inference_node:
  ros__parameters:
    # Backend configuration - TensorRT
    Backend:
      plugin: "onnxruntime_cpu"
```

### `deep_ort_gpu_backend_plugin`
Nvidia libraries must be installed separately alongside this plugin. Once installed, `deep_ort_gpu_backend_plugin` will automatically link to the nvidia libraries at runtime.

#### Prerequisites
Currently, `deep_ort_gpu_backend_plugin` supports the following nvidia configurations.

**TensorRT**: 10.9
**CUDA and CuDNN**: 12.0 to 12.8

List of compatible `nvidia/cuda` images:
- "12.8.0-cudnn-runtime-ubuntu22.04"
- "12.6.2-cudnn-runtime-ubuntu22.04"
- "12.5.1-cudnn-runtime-ubuntu22.04"
- "12.4.1-cudnn-runtime-ubuntu22.04"
- "12.3.2-cudnn-runtime-ubuntu22.04"
- "12.2.2-cudnn8-runtime-ubuntu22.04"
- "12.1.1-cudnn8-runtime-ubuntu22.04"
- "12.0.1-cudnn8-runtime-ubuntu22.04"

To download the minimal libraries needed for TensorRT:

```bash
curl -fsSL -o cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer10=10.9.0.34-1+cuda12.8 \
    libnvinfer-plugin10=10.9.0.34-1+cuda12.8 \
    libnvonnxparsers10=10.9.0.34-1+cuda12.8 \
    && rm cuda-keyring_1.1-1_all.deb
```

Note that `12.8` is compatible with all CUDA versions `12.0` to `12.8`
