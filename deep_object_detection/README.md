# Deep Object Detection Node

A deep learning object detection node for ROS 2. Simply provide any ONNX-compatible object detection model, and the node will automatically detect and adapt to its output format.

## Overview

The `deep_object_detection` package provides:
- Model-agnostic object detection using ONNX models
- Automatic output format detection and adaptation
- Multi-camera support with flexible batching (min/max batch size, timeout)
- Configurable preprocessing and postprocessing pipelines
- Explicit execution provider selection (TensorRT, CUDA, CPU) with fail-fast behavior
- Configurable failure policies (queue overflow, decode failure)
- Full ROS 2 lifecycle node support

## Package Contents

### Nodes

- **`deep_object_detection_node`**: A lifecycle node that subscribes to image messages (single or multi-camera), runs object detection inference, and publishes detection results.

### Configuration

- **`config/generic_model_params.yaml`**: Example configuration showing the parameter structure for model, preprocessing, postprocessing, and execution provider settings.

### Launch Files

- **`launch/deep_object_detection.launch.yaml`**: Launch file for the object detection node with lifecycle management.

## Usage

### Building the Package

```bash
cd /path/to/your/ros2_workspace
colcon build --packages-select deep_object_detection
source install/setup.bash
```

### Running the Node

1. **Basic usage**:

   ```bash
   ros2 launch deep_object_detection deep_object_detection.launch.yaml \
     config_file:=/path/to/generic_model_params.yaml
   ```

2. **With specific model path**:

   ```bash
   ros2 launch deep_object_detection deep_object_detection.launch.yaml \
     config_file:=/path/to/config.yaml \
     preferred_provider:=tensorrt
   ```

3. **Manual lifecycle transitions** (if not using lifecycle manager):

   ```bash
   # Configure the node
   ros2 lifecycle set /deep_object_detection_node configure

   # Activate the node
   ros2 lifecycle set /deep_object_detection_node activate
   ```

### Running with Camera Sync Node

The object detection node works with the `camera_sync` node for synchronized multi-camera processing.

1. **Launch camera sync node**:

   ```bash
   ros2 launch camera_sync multi_camera_sync.launch.yaml
   ```

2. **Launch object detection node**:

   ```bash
   ros2 launch deep_object_detection deep_object_detection.launch.yaml \
     config_file:=/path/to/config.yaml
   ```

   Configure the detection node to subscribe to the MultiImage topic:

   ```yaml
   camera_sync_topic: "/multi_camera_sync/multi_image_compressed"
   use_camera_sync: true
   min_batch_size: 3    # Minimum images before processing (match number of cameras)
   max_batch_size: 3    # Maximum images per batch
   ```

### Zero-Copy Component Container (High Performance)

For maximum performance, run both `multi_camera_sync` and `deep_object_detection` nodes in a single component container with intra-process communication:

```yaml
launch:
  - node_container:
      pkg: rclcpp_components
      exec: component_container_mt
      name: detection_container
      param:
        - name: use_intra_process_comms
          value: true
      composable_node:
        - pkg: camera_sync
          plugin: camera_sync::MultiCameraSyncNode
          name: multi_camera_sync
          extra_arg:
            - name: use_intra_process_comms
              value: "true"
        - pkg: deep_object_detection
          plugin: deep_object_detection::DeepObjectDetectionNode
          name: deep_object_detection_node
          extra_arg:
            - name: use_intra_process_comms
              value: "true"
```

## Configuration

The node uses a nested parameter structure for configuration:

```yaml
deep_object_detection_node:
  ros__parameters:
    # Required
    model_path: "/path/to/model.onnx"

    # Model configuration
    model:
      num_classes: 80
      bbox_format: "cxcywh"  # cxcywh, xyxy, or xywh

    # Preprocessing
    preprocessing:
      input_width: 640
      input_height: 640
      normalization_type: "scale_0_1"  # scale_0_1, imagenet, custom, none
      resize_method: "letterbox"  # letterbox, resize, crop, pad
      color_format: "rgb"  # rgb or bgr

    # Postprocessing
    postprocessing:
      score_threshold: 0.25
      nms_iou_threshold: 0.45
      max_detections: 300

    # Execution provider
    preferred_provider: "tensorrt"  # tensorrt, cuda, or cpu
    device_id: 0
    warmup_tensor_shapes: true
    enable_trt_engine_cache: true
    trt_engine_cache_path: "/tmp/deep_ros_ort_trt_cache"

    # Input/Output
    use_camera_sync: true
    camera_sync_topic: "/multi_camera_sync/multi_image_compressed"
    camera_topics: []  # List of individual camera topics (used when use_camera_sync: false)
    output_detections_topic: "/detections"
    min_batch_size: 1        # Minimum images before processing
    max_batch_size: 3        # Maximum images per batch
    max_batch_latency_ms: 0  # Max wait time in ms before processing (0 = wait for min_batch_size)
    queue_size: 10
    queue_overflow_policy: "drop_oldest"  # drop_oldest, drop_newest, or throw
    decode_failure_policy: "drop"         # drop or throw
```

### Execution Provider Selection

The node supports explicit provider selection with fail-fast behavior. If the specified provider is unavailable or fails to initialize, the node will immediately throw an error (no silent fallbacks).

**Available Providers:**
- `tensorrt` - TensorRT execution provider (requires CUDA and TensorRT)
- `cuda` - CUDA execution provider (requires CUDA)
- `cpu` - CPU execution provider (always available)

**Fail-Fast Behavior:**
- If `preferred_provider` is set to `tensorrt` but TensorRT is unavailable → node fails immediately
- If `preferred_provider` is set to `cuda` but CUDA runtime is unavailable → node fails immediately
- If model loading fails → node fails immediately
- No automatic fallback to other providers

## Expected Model Format

The node automatically detects and adapts to various ONNX model output formats:

- `[batch, detections, features]` - standard format
- `[batch, features, detections]` - transposed format
- `[batch, queries, 4+classes]` - query-based models (e.g., DETR)
- `[batch, channels, anchors]` - anchor-based models
- Any other layout - automatically detected and handled

**Input:** The node expects compressed images (sensor_msgs/CompressedImage) that will be decoded and preprocessed according to the configuration.

**Output:** Detection2DArray messages containing bounding boxes, scores, and class IDs.

## Parameters

### Required Parameters
- `model_path` (string): Path to ONNX model file

### Optional Parameters

**Model Configuration:**
- `class_names_path` (string): Path to class names file (one per line)
- `model.num_classes` (int, default: 80): Number of detection classes
- `model.bbox_format` (string, default: "cxcywh"): Bounding box format: "cxcywh", "xyxy", or "xywh"

**Preprocessing:**
- `preprocessing.input_width` (int, default: 640): Model input width
- `preprocessing.input_height` (int, default: 640): Model input height
- `preprocessing.normalization_type` (string, default: "scale_0_1"): "scale_0_1", "imagenet", "custom", or "none"
- `preprocessing.resize_method` (string, default: "letterbox"): "letterbox", "resize", "crop", or "pad"
- `preprocessing.color_format` (string, default: "rgb"): "rgb" or "bgr"

**Postprocessing:**
- `postprocessing.score_threshold` (float, default: 0.25): Minimum detection confidence
- `postprocessing.nms_iou_threshold` (float, default: 0.45): NMS IoU threshold
- `postprocessing.max_detections` (int, default: 300): Maximum detections per image (enforced after NMS)
- `postprocessing.score_activation` (string, default: "sigmoid"): Score activation: "sigmoid", "softmax", or "none"
- `postprocessing.enable_nms` (bool, default: true): Enable non-maximum suppression
- `postprocessing.class_score_mode` (string, default: "all_classes"): "all_classes" or "single_confidence"
- `postprocessing.coordinate_space` (string, default: "preprocessed"): "preprocessed" or "original"

**Execution Provider:**
- `preferred_provider` (string, default: "tensorrt"): Execution provider: "tensorrt", "cuda", or "cpu"
- `device_id` (int, default: 0): GPU device ID (for tensorrt/cuda)
- `warmup_tensor_shapes` (bool, default: true): Pre-warm tensor shape cache
- `enable_trt_engine_cache` (bool, default: true): Enable TensorRT engine caching
- `trt_engine_cache_path` (string, default: "/tmp/deep_ros_ort_trt_cache"): TensorRT cache directory

**Input/Output:**
- `camera_sync_topic` (string): MultiImage topic from camera sync node
- `camera_topics` (list of strings, default: []): Individual compressed image topics (used when `use_camera_sync: false`)
- `use_camera_sync` (bool, default: false): Use camera sync node (auto-enabled if `camera_sync_topic` is set)
- `output_detections_topic` (string, default: "/detections"): Output topic for detections
- `min_batch_size` (int, default: 1): Minimum images before processing (1 = process immediately)
- `max_batch_size` (int, default: 3): Maximum images per batch
- `max_batch_latency_ms` (int, default: 0): Maximum wait time in milliseconds before processing (0 = wait for `min_batch_size`)
- `queue_size` (int, default: 10): Image queue capacity
- `queue_overflow_policy` (string, default: "drop_oldest"): Behavior when queue is full: "drop_oldest", "drop_newest", or "throw"
- `decode_failure_policy` (string, default: "drop"): Behavior on image decode failure: "drop" or "throw"
- `input_qos_reliability` (string, default: "best_effort"): QoS reliability: "best_effort" or "reliable"

## Topics

### Input Topics
- `camera_sync_topic` (when `use_camera_sync: true`): MultiImage messages from camera sync node (contains synchronized compressed images)
- `camera_topics` (when `use_camera_sync: false`): Individual compressed image topics (sensor_msgs/CompressedImage)

**Note:** The node only supports compressed images. Raw images are not supported.

### Output Topics
- `output_detections_topic`: Detection2DArray messages (default: "/detections")

## Services (via Lifecycle)

Standard ROS 2 lifecycle services for state management:
- `configure`: Initialize the node and load the model
- `activate`: Start processing images
- `deactivate`: Stop processing images
- `cleanup`: Clean up resources
- `shutdown`: Shutdown the node

## Batching Behavior

The node supports flexible batching with three parameters:

- **`min_batch_size`**: Minimum number of images required before processing (default: 1)
- **`max_batch_size`**: Maximum number of images per batch (default: 3)
- **`max_batch_latency_ms`**: Maximum wait time in milliseconds before processing even if `min_batch_size` is not met (default: 0 = wait indefinitely)

**Use Cases:**
- **Single camera**: `min_batch_size=1, max_batch_size=1` - process each image immediately
- **Multi-camera synchronized**: `min_batch_size=N, max_batch_size=N` where N = number of cameras
- **Best-effort batching**: `min_batch_size=1, max_batch_size=N` with optional `max_batch_latency_ms` timeout

## Failure Policies

The node provides configurable failure handling for robustness:

**Queue Overflow Policy** (`queue_overflow_policy`):
- `drop_oldest` (default): Remove oldest image from queue, enqueue new image
- `drop_newest`: Drop the new image, keep queue unchanged
- `throw`: Throw exception (fail-fast mode)

**Decode Failure Policy** (`decode_failure_policy`):
- `drop` (default): Log warning and drop the frame
- `throw`: Throw exception (fail-fast mode)

By default, both policies use "drop" behavior, making the node resilient to occasional failures without crashing.

## Dependencies

- `deep_core`: Core inference framework
- `deep_conversions`: ROS message to tensor conversions
- `deep_ort_backend_plugin`: ONNX Runtime backend plugin
- `deep_msgs`: Custom ROS messages (MultiImage for camera sync)
- `rclcpp_lifecycle`: ROS 2 lifecycle management
- `sensor_msgs`: For compressed image message types
- `nav2_lifecycle_manager`: For automatic lifecycle transitions (optional)
- `camera_sync`: For multi-camera synchronization (optional)
- OpenCV: For image preprocessing and decoding
- ONNX Runtime: For model inference (CPU/CUDA/TensorRT)

## Troubleshooting

- **"No plugin loaded"**: Check that the backend plugin name is correct in the configuration
- **"No model loaded"**: Verify the model path exists and is a valid ONNX file
- **"Provider initialization failed"**: Check that the specified execution provider (tensorrt/cuda) is available and properly configured
- **Lifecycle errors**: Ensure the node is properly configured before activation
- **No detections**: Verify that the model output format matches the expected format, or enable auto-detection
- **Plugin discovery issues**: Check that `deep_ort_backend_plugin` is built and sourced
- **CUDA/TensorRT errors**: Ensure CUDA libraries and TensorRT (if using) are properly installed and accessible
