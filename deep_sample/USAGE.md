# Deep Sample Usage Guide

This guide shows how to use the `deep_sample` package to run inference with the deep_core framework.

## Quick Start

### 1. Build the workspace

```bash
cd /workspaces/deep_ros
colcon build --packages-select deep_sample
source install/setup.bash
```

### 2. Prepare a model (optional)
If you have PyTorch available:

```bash
cd deep_sample
python3 scripts/create_simple_model.py
```

Or place your own ONNX model in `deep_sample/models/simple_model.onnx`

### 3. Launch the node

```bash
ros2 launch deep_sample sample_inference.launch.py
```

### 4. Manage the lifecycle (in separate terminals)

```bash
# Configure the node (loads plugin and model)
ros2 lifecycle set /sample_inference_node configure

# Activate the node (starts processing)
ros2 lifecycle set /sample_inference_node activate

# Check status
ros2 lifecycle get /sample_inference_node
```

### 5. Test with dummy data

```bash
# Monitor the output
ros2 topic echo /inference/output

# Send a test image (in another terminal)
ros2 topic pub --once /camera/image_raw sensor_msgs/msg/Image '{
  header: {frame_id: "camera"},
  height: 224,
  width: 224,
  encoding: "rgb8",
  step: 672,
  data: [128]
}'
```

## Configuration

The configuration uses the nested Backend structure:

```yaml
sample_inference_node:
  ros__parameters:
    Backend:
      plugin: "deep_ort_backend_plugin::OrtBackendPlugin"
    model_path: "/path/to/your/model.onnx"
    input_topic: "/camera/image_raw"
    output_topic: "/inference/output"
```

## Customizing the Example

### Using your own model
1. Place your ONNX model in the `models/` directory
2. Update the `model_path` parameter in the configuration
3. Ensure your model expects input shape `[1, 3, 224, 224]`

### Changing topics
Update the `input_topic` and `output_topic` parameters:

```bash
ros2 launch deep_sample sample_inference.launch.py \
  input_topic:=/my_camera/image \
  output_topic:=/my_inference/result
```

### Using different backends
The framework supports different backend plugins. To use a different one:

```yaml
Backend:
  plugin: "your_backend_plugin::YourBackendPlugin"
```

## Lifecycle States

The node supports these lifecycle states:

1. **Unconfigured** → `configure` → **Inactive**
2. **Inactive** → `activate` → **Active**
3. **Active** → `deactivate` → **Inactive**
4. **Inactive** → `cleanup` → **Unconfigured**

Only in the **Active** state will the node process images and run inference.

## Troubleshooting

### "No plugin loaded"
- Check the plugin name in the configuration
- Ensure `deep_ort_backend_plugin` is built and sourced

### "No model loaded"
- Verify the model path exists
- Check that the file is a valid ONNX model
- Ensure the node is configured before activation

### "Inference failed"
- Check model input/output shapes match expectations
- Verify the model is compatible with ONNX Runtime
- Check logs for detailed error messages

### Plugin discovery issues

```bash
# List available plugins
ros2 pkg list | grep deep

# Check if plugins are registered
ros2 run pluginlib_tutorials list_plugins deep_core
```

## Integration with real cameras

To use with real camera data:

1. **USB camera**:

   ```bash
   ros2 run usb_cam usb_cam_node_exe
   ```

2. **Realsense camera**:

```bash
   ros2 launch realsense2_camera rs_launch.py
   ```

1. **Update topic mapping**:

   ```bash
ros2 launch deep_sample sample_inference.launch.py \
     input_topic:=/camera/color/image_raw

   ```
