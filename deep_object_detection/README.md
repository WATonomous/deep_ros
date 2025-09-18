# Deep Object Detection

A ROS 2 package for real-time object detection using ONNX Runtime with GPU acceleration.

## Overview

This package provides an abstracted inference interface that supports ONNX models with GPU acceleration via the `onnxruntime_gpu_vendor` package. The design abstracts away the inference engine implementation, making it easy to switch between different inference backends.

## Features

- **TensorRT Optimization**: High-speed inference using TensorRT engine files
- **Batched Processing**: Process multiple images simultaneously for improved throughput
- **Multi-Camera Support**: Subscribe to multiple camera topics with configurable batch sizes
- **Model Agnostic**: Works with any TensorRT-compatible object detection model (YOLOv8, YOLOv5, etc.)
- **Real-time Visualization**: Optional RViz markers for detection visualization
- **Performance Monitoring**: Built-in performance statistics and debugging

## Requirements

- ROS2 (Humble/Iron recommended)
- CUDA-capable GPU
- TensorRT 8.x+
- OpenCV 4.x
- yaml-cpp

## Installation

1. Install dependencies:

```bash
cd ~/deep_ros
rosdep install --from-paths src --ignore-src -r -y
```

1. Build the package:

```bash
colcon build --packages-select deep_object_detection
```

1. Source the workspace:

```bash
source ~/deep_ros/install/setup.bash
```

## Usage

### 1. Prepare Your Model

Convert your object detection model to a TensorRT engine:

```bash
# Example for YOLOv8 (adjust paths and parameters as needed)
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```

### 2. Configure the Node

Edit the configuration file at `config/object_detection_config.yaml`:

```yaml
# Update these paths and parameters for your setup
model_engine_path: "/path/to/your/model.engine"
camera_topics:
  - "/camera1/image_raw"
  - "/camera2/image_raw"
max_batch_size: 4
confidence_threshold: 0.5
class_names: ["person", "car", "bicycle", ...]  # Your model's classes
```

### 3. Launch the Node

#### Single Camera:

```bash
ros2 launch deep_object_detection object_detection.launch.py \
    model_engine_path:=/path/to/your/model.engine \
    camera_topics:='["/camera/image_raw"]'
```

#### Multiple Cameras:

```bash
ros2 launch deep_object_detection multi_camera_detection.launch.py \
    model_engine_path:=/path/to/your/model.engine \
    max_batch_size:=8
```

#### With Custom Parameters:

```bash
ros2 run deep_object_detection object_detection_node \
    --ros-args \
    -p model_engine_path:=/path/to/model.engine \
    -p camera_topics:='["/camera1/image_raw", "/camera2/image_raw"]' \
    -p max_batch_size:=6 \
    -p confidence_threshold:=0.6 \
    -p enable_visualization:=true
```

## Topics

### Subscribed Topics
- `/camera*/image_raw` (sensor_msgs/Image): Input camera images

### Published Topics
- `/detections` (vision_msgs/Detection2DArray): Detection results
- `/detection_markers` (visualization_msgs/MarkerArray): Visualization markers for RViz
- `/performance_stats` (std_msgs/String): Performance statistics (when debug enabled)

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_engine_path` | string | - | Path to TensorRT engine file |
| `camera_topics` | string[] | ["/camera/image_raw"] | Camera topics to subscribe to |
| `max_batch_size` | int | 4 | Maximum batch size for inference |
| `inference_rate` | double | 30.0 | Target inference rate (Hz) |
| `confidence_threshold` | double | 0.5 | Detection confidence threshold |
| `nms_threshold` | double | 0.4 | Non-maximum suppression threshold |
| `input_width` | int | 640 | Model input width |
| `input_height` | int | 640 | Model input height |
| `enable_visualization` | bool | true | Enable RViz markers |
| `enable_debug` | bool | false | Enable debug output |
| `class_names` | string[] | COCO classes | Object class names |

## Performance Optimization

### Batch Size Tuning
- Larger batch sizes improve GPU utilization but increase latency
- Start with batch size 4-8 and adjust based on your requirements
- Monitor GPU memory usage and inference times

### Model Optimization
- Use FP16 precision for 2x speed improvement with minimal accuracy loss
- Optimize your TensorRT engine for your specific GPU architecture
- Consider dynamic shape optimization for variable input sizes

### System Configuration
- Ensure adequate GPU memory (8GB+ recommended)
- Use high-bandwidth camera interfaces (USB 3.0+, GigE)
- Pin node to specific CPU cores for consistent performance

## Example Output

```bash
[INFO] [object_detection_node]: TensorRT inference engine initialized successfully
[INFO] [object_detection_node]: Subscribed to camera topic: /camera1/image_raw
[INFO] [object_detection_node]: Subscribed to camera topic: /camera2/image_raw
[DEBUG] [object_detection_node]: Batch inference time: 15 ms, batch size: 4
```

## Troubleshooting

### Common Issues

1. **TensorRT Engine Loading Failed**
   - Check engine path and file permissions
   - Ensure engine was built for correct GPU architecture
   - Verify TensorRT version compatibility

2. **Low Performance**
   - Check GPU utilization with `nvidia-smi`
   - Increase batch size if GPU utilization is low
   - Verify camera frame rates match inference rate

3. **Memory Issues**
   - Reduce batch size or model input resolution
   - Check available GPU memory
   - Monitor system RAM usage

### Debug Mode
Enable debug mode for detailed performance metrics:

```bash
ros2 param set /object_detection_node enable_debug true
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera 1      │    │                  │    │   Detections    │
│   /image_raw    ├────┤  Object Detection│    │   Publisher     │
└─────────────────┘    │      Node        ├────┤                 │
┌─────────────────┐    │                  │    └─────────────────┘
│   Camera 2      │    │  - Batch Images  │    ┌─────────────────┐
│   /image_raw    ├────┤  - TensorRT      │    │  Visualization  │
└─────────────────┘    │  - Post-process  ├────┤   Markers       │
┌─────────────────┐    │  - NMS           │    │                 │
│   Camera N      │    │                  │    └─────────────────┘
│   /image_raw    ├────┤                  │
└─────────────────┘    └──────────────────┘
```

## License

[Your License Here]

## Contributing

Please read the contributing guidelines before submitting pull requests.
