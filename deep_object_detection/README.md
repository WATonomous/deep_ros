# Deep Object Detection Node

A **completely model-agnostic** deep learning object detection node for ROS2. Simply provide any ONNX-compatible object detection model, and the node will automatically detect and adapt to its output format. No model-specific configuration needed!

## Features

- **Fully Model-Agnostic**: Works with **ANY** ONNX detection model - automatically detects and adapts to the output format
- **Zero Configuration**: Just provide the model path - the node figures out the rest
- **Auto-Detection**: Automatically detects output tensor layout, coordinate format, and feature locations
- **Configurable Preprocessing**: Multiple normalization schemes (ImageNet, scale_0_1, custom, none)
- **Multiple Resize Methods**: Letterbox, simple resize, crop, pad
- **Multi-Camera Support**: Batched processing from multiple camera inputs
- **Lifecycle Management**: Full ROS2 lifecycle node support
- **Provider Fallback**: Automatic fallback from TensorRT → CUDA → CPU

## How It Works

The node uses a **generic postprocessor** that automatically:

1. **Detects the output tensor shape** from your model
2. **Infers the layout** (which dimensions are batch, detections, features)
3. **Extracts bounding boxes** based on detected format (cxcywh, xyxy, xywh)
4. **Finds scores and class IDs** from the output tensor
5. **Applies NMS and coordinate transformation** to original image space

**No model-specific code needed!** Just provide your ONNX model and it works.

### Supported Output Formats (Auto-Detected)

The generic postprocessor automatically adapts to **any** tensor layout:
- `[batch, detections, features]` - standard format
- `[batch, features, detections]` - transposed format
- `[batch, queries, 4+classes]` - query-based models
- `[batch, channels, anchors]` - anchor-based models
- Any other layout - automatically detected and handled

## Configuration

### Model Metadata (Minimal - Auto-Detected)

```yaml
model:
  num_classes: 80  # Number of classes your model detects
  bbox_format: "cxcywh"  # cxcywh, xyxy, xywh (auto-detected if not specified)
```

### Preprocessing

```yaml
preprocessing:
  input_width: 640   # Match your model's expected input size
  input_height: 640
  normalization_type: "scale_0_1"  # scale_0_1, imagenet, custom, none
  resize_method: "letterbox"  # letterbox, resize, crop, pad
  color_format: "rgb"  # rgb or bgr
```

### Postprocessing

```yaml
postprocessing:
  score_threshold: 0.25
  nms_iou_threshold: 0.45
  max_detections: 300
```

## Usage

### Basic Usage (Model-Agnostic - Recommended)

Simply provide your ONNX model path:

```bash
ros2 launch deep_object_detection deep_object_detection.launch.py \
  config_file:=/path/to/generic_model_params.yaml
```

The node will automatically detect and adapt to your model's output format!

### Running with Camera Sync Node

The object detection node works seamlessly with the `camera_sync` node for synchronized multi-camera processing. Here's how to run them together:

#### Step 1: Launch Camera Sync Node

First, launch the camera sync node to synchronize your camera feeds:

```bash
ros2 launch camera_sync multi_camera_sync.launch.yaml
```

Or with custom configuration using a parameter file:

Create a config file `camera_sync_config.yaml`:

```yaml
multi_camera_sync:
  ros__parameters:
    camera_topics:
      - "/CAM_FRONT/image_rect_compressed"
      - "/CAM_FRONT_LEFT/image_rect_compressed"
      - "/CAM_FRONT_RIGHT/image_rect_compressed"
    use_compressed: true
    sync_tolerance_ms: 33.0
```

Then launch with:

```bash
ros2 launch camera_sync multi_camera_sync.launch.yaml \
  --ros-args --params-file camera_sync_config.yaml
```

Alternatively, modify the default config file at:
`$(find-pkg-share camera_sync)/config/multi_camera_sync_params.yaml`

#### Step 2: Configure Detection Node

The object detection node subscribes to the **MultiImage topic** published by the camera sync node, not individual camera topics. Configure it like this:

```yaml
camera_sync_topic: "/multi_camera_sync/multi_image_compressed"
camera_topics: []  # Leave empty when using camera sync
batch_size_limit: 3  # Should match number of cameras being synchronized
```

**How it works:**
- Camera sync node subscribes to individual camera topics (e.g., `/CAM_FRONT/image_rect_compressed`, etc.)
- Camera sync synchronizes them and publishes a **single MultiImage message** containing all synchronized images
- Object detection node subscribes to that MultiImage topic
- When a MultiImage message arrives, the detection node extracts all images and processes them as a batch

#### Step 3: Launch Detection Node

Launch the object detection node (it will subscribe to the synchronized camera topics):

```bash
ros2 launch deep_object_detection deep_object_detection.launch.py \
  config_file:=/workspaces/deep_ros/deep_object_detection/config/generic_model_params.yaml
```

#### Complete Example: Running Both Nodes Together

You can launch both nodes in a single command:

```bash
# Terminal 1: Launch camera sync
ros2 launch camera_sync multi_camera_sync.launch.yaml

# Terminal 2: Launch object detection (after camera sync is running)
ros2 launch deep_object_detection deep_object_detection.launch.py \
  config_file:=/workspaces/deep_ros/deep_object_detection/config/generic_model_params.yaml
```

Or create a combined launch file to run both:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Camera sync node
        Node(
            package='camera_sync',
            executable='multi_camera_sync_node',
            name='multi_camera_sync',
            parameters=[{
                'camera_topics': [
                    '/CAM_FRONT/image_rect_compressed',
                    '/CAM_FRONT_LEFT/image_rect_compressed',
                    '/CAM_FRONT_RIGHT/image_rect_compressed'
                ],
                'use_compressed': True,
                'sync_tolerance_ms': 33.0,
            }]
        ),
        # Object detection node
        Node(
            package='deep_object_detection',
            executable='deep_object_detection_node',
            name='deep_object_detection_node',
            parameters=[{
                'model_path': '/workspaces/deep_ros/yolov8m.onnx',
                'camera_topics': [
                    '/CAM_FRONT/image_rect_compressed',
                    '/CAM_FRONT_LEFT/image_rect_compressed',
                    '/CAM_FRONT_RIGHT/image_rect_compressed'
                ],
                'batch_size_limit': 3,
            }]
        ),
    ])
```

#### How It Works Together

1. **Camera Sync Node**:
   - Subscribes to individual camera topics (e.g., `/CAM_FRONT/image_rect_compressed`, `/CAM_FRONT_LEFT/image_rect_compressed`, etc.)
   - Uses message filters to synchronize images based on timestamps
   - Publishes a **single MultiImage message** to `~/multi_image_compressed` containing all synchronized images as an array

2. **Object Detection Node**:
   - Subscribes to the MultiImage topic from camera sync (e.g., `/multi_camera_sync/multi_image_compressed`)
   - When a MultiImage message arrives, extracts all images from the array
   - Enqueues all images from the synchronized batch
   - Processes them together through the model

3. **Data Flow**:

   ```
Camera Topics → Camera Sync Node → MultiImage Message → Detection Node → Detections

   ```

**Key Point**: The camera sync node publishes **one topic** (`~/multi_image_compressed`) that contains **all synchronized images** in a single message. The detection node subscribes to this topic and extracts the individual images for batch processing.

### Minimal Configuration

The node requires minimal configuration - just the model path and basic settings:

```yaml
model_path: "/path/to/your/model.onnx"
class_names_path: "/path/to/classes.txt"
preprocessing:
  input_width: 640
  input_height: 640
postprocessing:
  score_threshold: 0.25
  nms_iou_threshold: 0.45
```

Everything else is auto-detected from your model!

## Backward Compatibility

The node supports backward compatibility with legacy parameter names:
- `input_width/input_height` → `preprocessing.input_width/input_height`
- `use_letterbox` → `preprocessing.resize_method: letterbox`
- `score_threshold` → `postprocessing.score_threshold`
- `preboxed_format` → `model.bbox_format`
- `normalization_type: "yolo"` → `normalization_type: "scale_0_1"` (same behavior)

## API Reference

### Input Topics
- `camera_topics`: List of compressed image topics for multi-camera input

### Output Topics
- `output_detections_topic`: Detection2DArray messages

### Services (via Lifecycle)
- Standard ROS2 lifecycle services for state management

## Dependencies

- ROS2 (Humble/Iron)
- ONNX Runtime (CPU/CUDA/TensorRT)
- OpenCV
- deep_core, deep_msgs
