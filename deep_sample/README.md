# Deep Sample Package

This package demonstrates how to use the `deep_core` framework with the `deep_ort_backend_plugin` for deep learning inference in ROS 2.

## Overview

The `deep_sample` package provides a simple example of:
- Extending `DeepNodeBase` to create custom inference nodes
- Loading ONNX models using the ONNX Runtime backend plugin
- Using `deep_conversions` to convert ROS messages to tensors
- Processing image data and publishing inference results
- Proper lifecycle management for inference nodes

## Package Contents

### Nodes

- **`sample_inference_node`**: A lifecycle node that subscribes to image messages, runs inference, and publishes results.

### Configuration

- **`config/sample_config.yaml`**: Example configuration showing the nested parameter structure for backend plugin selection.

### Launch Files

- **`launch/sample_inference.launch.py`**: Basic launch file for the inference node.
- **`launch/sample_with_lifecycle.launch.py`**: Advanced launch file with lifecycle management options.

### Models

- **`models/`**: Directory for ONNX model files.
- **`scripts/`**: Scripts to create simple test models.

## Usage

### Building the Package

```bash
cd /path/to/your/ros2_workspace
colcon build --packages-select deep_sample
source install/setup.bash
```

### Preparing a Model

1. Place an ONNX model in the `models/` directory, or
2. Create a simple test model using the provided scripts:

```bash
   cd deep_sample
   python3 scripts/create_simple_model.py  # Requires PyTorch
   # or
   python3 scripts/create_dummy_model.py   # Requires ONNX
   ```

### Running the Node

1. **Basic usage** (requires manual lifecycle transitions):

   ```bash
   ros2 launch deep_sample sample_inference.launch.py
   ```

2. **With specific model path**:

```bash
   ros2 launch deep_sample sample_inference.launch.py model_path:=/path/to/your/model.onnx
   ```

1. **Manual lifecycle transitions** (in separate terminals):

   ```bash
# Configure the node
   ros2 lifecycle set /sample_inference_node configure

# Activate the node
   ros2 lifecycle set /sample_inference_node activate

   ```

### Testing the Node

1. **Check node status**:

   ```bash
   ros2 lifecycle get /sample_inference_node
   ```

1. **Monitor topics**:

```bash
   # List topics
   ros2 topic list

   # Monitor output
   ros2 topic echo /inference/output
   ```

1. **Publish test images** (if you have image data):

   ```bash
ros2 topic pub /camera/image_raw sensor_msgs/msg/Image '{
     header: {frame_id: "camera"},
     height: 480,
     width: 640,
     encoding: "rgb8",
     step: 1920,
     data: [...]
   }'

   ```

## Configuration

The node uses the nested parameter structure for backend plugin selection:

```yaml
sample_inference_node:
  ros__parameters:
    Backend:
      plugin: "deep_ort_backend_plugin::OrtBackendPlugin"
    model_path: "/path/to/model.onnx"
    input_topic: "/camera/image_raw"
    output_topic: "/inference/output"
```

## Expected Model Format

The sample node expects ONNX models with:
- **Input**: `[1, 3, 224, 224]` (batch_size, channels, height, width)
- **Output**: `[1, N]` where N is the number of output features

## Extending the Example

To create your own inference node:

1. **Inherit from `DeepNodeBase`**:

   ```cpp
   class MyInferenceNode : public deep_ros::DeepNodeBase
   {
     // Implement lifecycle callbacks
   };
   ```

2. **Implement the lifecycle callbacks**:
   - `on_configure_impl()`: Set up subscribers/publishers
   - `on_activate_impl()`: Start processing
   - `on_deactivate_impl()`: Stop processing
   - `on_cleanup_impl()`: Clean up resources

3. **Use deep_conversions for message handling**:

   ```cpp
// Convert ROS image to tensor
   auto allocator = get_current_allocator();
   auto input_tensor = deep_ros::ros_conversions::from_image(*msg, allocator);

   // Run inference
   auto output_tensor = run_inference(input_tensor);

   ```

## Dependencies

- `deep_core`: Core inference framework
- `deep_conversions`: ROS message to tensor conversions
- `deep_ort_backend_plugin`: ONNX Runtime backend
- `rclcpp_lifecycle`: ROS 2 lifecycle management
- `sensor_msgs`: For image message types
- `std_msgs`: For output message types

## Troubleshooting

- **"No plugin loaded"**: Check that the backend plugin name is correct in the configuration
- **"No model loaded"**: Verify the model path exists and is a valid ONNX file
- **Lifecycle errors**: Ensure the node is properly configured before activation
- **Plugin discovery issues**: Check that `deep_ort_backend_plugin` is built and sourced
