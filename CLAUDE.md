You have permission as claude to edit the claude.md file to keep it up to date with our current conversations.

# Deep ROS - ML Infrastructure Pipeline

## Project Overview
Deep ROS is an open-source ML infrastructure pipeline that enables users to train models, quantize them, and deploy them on ROS nodes. The core paradigm is creating **generic ROS nodes that act as containers** for quantized models, rather than nodes with pre-built models.

## Architecture Philosophy

### Generic ROS Node Containers
- ROS nodes are model-agnostic containers that can load any compatible quantized model
- Nodes provide standardized input/output interfaces for model inference
- Models are loaded at runtime, not compiled into the node

### Benefits
- **Stable Dependency Tree**: ROS node dependencies remain consistent regardless of model changes
- **Flexible Deployment**: Same node can run different models without recompilation
- **Build Farm Efficiency**: Model building handled separately with support for multiple environments/versions

## Directory Structure Plan
```
deep_ros/
├── model_farm/           # Model training/quantization (COLCON_IGNORE)
├── ros_nodes/           # Generic ROS node containers
├── launch/              # Launch files and configurations
└── interfaces/          # ROS message/service definitions
```

## Supported Model Formats
- **ONNX**: Primary format
- **TensorRT**: NVIDIA GPU optimization
- **OpenVINO**: Intel hardware optimization  
- **TensorFlow Lite**: Mobile/edge deployment
- **CoreML**: Apple hardware
- **RKNN**: Rockchip NPU models

## Distribution Strategy
- ROS nodes distributed as apt-installable packages
- Users can `apt install` specific node packages
- Point node to model file path in launch configuration
- Model files distributed separately from node packages

## Usage Pattern
1. Install desired ROS node: `apt install ros-<distro>-deep-inference-node`
2. Prepare quantized model (ONNX/TensorRT/etc.)
3. Configure launch file with model path
4. Launch node with model loaded at runtime

## Directory Structure (Current)
```
deep_ros/
├── deep_core/                     # Core components
│   ├── include/deep_core/
│   │   ├── types/
│   │   │   └── tensor.hpp        # Memory-safe tensor class
│   │   ├── deep_node_base.hpp    # Generic lifecycle ROS node base class
│   │   └── plugin_interface.hpp  # Pure plugin interface
│   └── src/
├── deep_backends/                 # Backend plugin packages
│   ├── deep_tensorrt_plugin/      # TensorRT backend plugin
│   ├── deep_openvino_plugin/      # OpenVINO backend plugin
│   ├── deep_onnxruntime_plugin/   # ONNX Runtime backend plugin
│   └── deep_tflite_plugin/        # TensorFlow Lite backend plugin
├── deep_msgs/                     # ROS message definitions
├── deep_bringup/                  # Launch files and configurations
├── deep_examples/                 # Example configurations
└── model_farm/                    # Training pipeline (COLCON_IGNORE)
```

## Plugin Architecture
- Generic lifecycle inference node (DeepNodeBase) manages plugin loading/unloading
- Users inherit from DeepNodeBase and override *_impl methods for custom behavior  
- Base class handles backend management, then calls user implementation
- Each backend plugin declares system dependencies via rosdep in package.xml
- Users install only the backend plugins they need
- Memory-safe tensor class with verbose error messages

## Implementation Status
- ✅ Core tensor class with memory safety and error handling
- ✅ Plugin interface with detailed error results
- ✅ DeepNodeBase lifecycle management with user override pattern
- ✅ Basic directory structure and headers

## TODO Items
- Plugin discovery using pluginlib (`discover_available_plugins()` in deep_node_base.cpp:165)
- Plugin loading using pluginlib class_loader (`load_plugin_library()` in deep_node_base.cpp:170)
- Define ROS message interfaces in deep_msgs/
- Create CMakeLists.txt and package.xml files
- Implement example backend plugins
- Set up model farm build pipeline

## Model Farm Design Notes
- Conversion from project models to ONNX is **left to implementers**
- We provide example conversion scripts but don't enforce specific methods
- Each project handles its own dependencies via Docker
- Infrastructure navigation in bash, specialized tasks in Python
- ROS CLI (deep_tools) only validates/checks ONNX files, no conversion