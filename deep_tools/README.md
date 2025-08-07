# (TODO) Deep Tools - ROS CLI for ONNX Model Management

ROS2 CLI tool for ONNX model validation and compatibility checking.

## Commands

```bash
ros2 model check model.onnx                    # Check compatibility with available ROS nodes
ros2 model validate model.onnx                 # Validate ONNX structure and integrity  
ros2 model info model.onnx                     # Show model I/O shapes, operations, metadata
ros2 model list-nodes                          # Show available Deep ROS node types
```

## Design Principles

- **No conversion**: Model conversion is implementation-specific and handled by model_farm
- **Framework-agnostic**: Only works with existing ONNX files
- **Pure inspection**: Validates and analyzes models, doesn't modify them
- **ROS integration**: Follows ROS2 CLI patterns like `ros2 bag`

## Architecture

- Compatibility checking via node specification files
- Generic ONNX validation using onnx library
- Model introspection for I/O analysis
- Integration with Deep ROS node ecosystem