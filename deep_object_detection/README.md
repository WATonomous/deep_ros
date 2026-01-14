# Deep Object Detection Node

A deep learning object detection node for ROS 2. Simply provide any ONNX-compatible object detection model, and the node will automatically detect and adapt to its output format.

## Overview

The `deep_object_detection` package provides:
- Model-agnostic object detection using ONNX models
- Automatic output format detection and adaptation
- Multi-camera support via MultiImage messages with flexible batching
- Configurable preprocessing and postprocessing pipelines
- Explicit execution provider selection (TensorRT, CUDA, CPU) with fail-fast behavior
- Simplified configuration with sensible defaults and hardcoded behaviors
- Full ROS 2 lifecycle node support

## Package Contents

### Nodes

- **`deep_object_detection_node`**: A lifecycle node that subscribes to MultiImage messages (synchronized multi-camera input), runs object detection inference, and publishes detection results.

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

The object detection node requires the `camera_sync` node for synchronized multi-camera processing.

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
   input_topic: "/multi_camera_sync/multi_image_compressed"
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

The node uses a nested parameter structure for configuration. Many parameters have sensible defaults, and some behaviors are hardcoded for simplicity.

### Configuration Example

```yaml
deep_object_detection_node:
  ros__parameters:
    # Required
    model_path: "/path/to/model.onnx"
    input_topic: "/multi_camera_sync/multi_image_compressed"

    # Model configuration
    model:
      num_classes: 80
      bbox_format: "cxcywh"  # cxcywh, xyxy, or xywh
    class_names_path: "/path/to/classes.txt"  # Optional

    # Preprocessing
    preprocessing:
      input_width: 640
      input_height: 640
      normalization_type: "scale_0_1"  # scale_0_1, imagenet, custom, none
      resize_method: "letterbox"  # letterbox, resize, crop, pad
      color_format: "rgb"  # rgb or bgr
      mean: [0.0, 0.0, 0.0]  # For custom normalization
      std: [1.0, 1.0, 1.0]   # For custom normalization

    # Postprocessing
    postprocessing:
      score_threshold: 0.25
      nms_iou_threshold: 0.45
      max_detections: 300
      score_activation: "sigmoid"  # sigmoid, softmax, none
      class_score_mode: "all_classes"  # all_classes or single_confidence

      # Multi-output model configuration (only needed if use_multi_output: true)
      use_multi_output: false
      output_boxes_idx: 0
      output_scores_idx: 1
      output_classes_idx: 2

      # Manual layout configuration (only needed if auto_detect: false)
      layout:
        auto_detect: true
        batch_dim: 0
        detection_dim: 1
        feature_dim: 2
        bbox_start_idx: 0
        bbox_count: 4
        score_idx: 4
        class_idx: 5

    # Execution provider
    preferred_provider: "tensorrt"  # tensorrt, cuda, or cpu
    device_id: 0
    enable_trt_engine_cache: false
    trt_engine_cache_path: "/tmp/deep_ros_ort_trt_cache"

    # Batching
    min_batch_size: 1
    max_batch_size: 3
    max_batch_latency_ms: 0  # 0 = wait for min_batch_size
    queue_size: 10  # 0 = unlimited

    # Output
    output_detections_topic: "/detections"
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

**Input:** The node expects MultiImage messages (deep_msgs/MultiImage) containing synchronized compressed images from multiple cameras. Images are decoded immediately in the subscription callback.

**Output:** Detection2DArray messages containing bounding boxes, scores, and class IDs for each image in the batch.

## Parameters

### Required Parameters
- **`model_path`** (string): Path to ONNX model file. Must be a valid ONNX model.
- **`input_topic`** (string): MultiImage topic name to subscribe to. The node only supports MultiImage messages (synchronized multi-camera input).

### Model Configuration

- **`model.num_classes`** (int, default: 80): Number of detection classes in the model. Used for postprocessing.
- **`model.bbox_format`** (string, default: "cxcywh"): Bounding box format used by the model.
  - `"cxcywh"`: Center x, center y, width, height (YOLO format)
  - `"xyxy"`: Top-left x, top-left y, bottom-right x, bottom-right y
  - `"xywh"`: Top-left x, top-left y, width, height
- **`class_names_path`** (string, optional): Path to text file with class names (one per line). If not provided, class IDs will be used in output messages.

### Preprocessing Configuration

- **`preprocessing.input_width`** (int, default: 640): Model input image width in pixels.
- **`preprocessing.input_height`** (int, default: 640): Model input image height in pixels.
- **`preprocessing.normalization_type`** (string, default: "scale_0_1"): Normalization method.
  - `"scale_0_1"`: Scale pixel values to [0, 1] range (divide by 255.0)
  - `"imagenet"`: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - `"custom"`: Use `preprocessing.mean` and `preprocessing.std` values
  - `"none"`: No normalization
- **`preprocessing.mean`** (list of 3 floats, default: [0.0, 0.0, 0.0]): Mean values for custom normalization (RGB order).
- **`preprocessing.std`** (list of 3 floats, default: [1.0, 1.0, 1.0]): Standard deviation values for custom normalization (RGB order).
- **`preprocessing.resize_method`** (string, default: "letterbox"): Image resizing method.
  - `"letterbox"`: Maintain aspect ratio, pad with gray (114) to fit model input size
  - `"resize"`: Stretch image to model input size (may distort aspect ratio)
  - `"crop"`: Crop image to model input size
  - `"pad"`: Pad image to model input size
- **`preprocessing.color_format`** (string, default: "rgb"): Expected color format.
  - `"rgb"`: RGB color format
  - `"bgr"`: BGR color format (OpenCV default)

**Hardcoded Assumptions:**
- **`preprocessing.pad_value`**: Always 114 (YOLO gray padding value). Not configurable.

### Postprocessing Configuration

- **`postprocessing.score_threshold`** (float, default: 0.25): Minimum confidence score for detections. Detections below this threshold are filtered out.
- **`postprocessing.nms_iou_threshold`** (float, default: 0.45): IoU threshold for Non-Maximum Suppression. Higher values allow more overlapping boxes.
- **`postprocessing.max_detections`** (int, default: 300): Maximum number of detections per image (enforced after NMS).
- **`postprocessing.score_activation`** (string, default: "sigmoid"): Activation function for raw scores.
  - `"sigmoid"`: Apply sigmoid activation (for YOLO-style models)
  - `"softmax"`: Apply softmax activation
  - `"none"`: Use raw scores
- **`postprocessing.class_score_mode`** (string, default: "all_classes"): How class scores are extracted.
  - `"all_classes"`: Extract class scores from all class logits (for multi-class models)
  - `"single_confidence"`: Use a single confidence score (for single-class models)

**Hardcoded Assumptions:**
- **`postprocessing.enable_nms`**: Always enabled. Not configurable.
- **`postprocessing.coordinate_space`**: Always `"preprocessed"` (coordinates are transformed from preprocessed image space to original image space). Not configurable.

**Conditional Parameters (only used when `use_multi_output: true`):**
- **`postprocessing.use_multi_output`** (bool, default: false): Enable multi-output model support (separate outputs for boxes, scores, classes).
- **`postprocessing.output_boxes_idx`** (int, default: 0): Output index for bounding boxes (only used if `use_multi_output: true`).
- **`postprocessing.output_scores_idx`** (int, default: 1): Output index for scores (only used if `use_multi_output: true`).
- **`postprocessing.output_classes_idx`** (int, default: 2): Output index for class IDs (only used if `use_multi_output: true`).

**Conditional Parameters (only used when `layout.auto_detect: false`):**
- **`postprocessing.layout.auto_detect`** (bool, default: true): Automatically detect output tensor layout. If `false`, manual layout parameters must be provided.
- **`postprocessing.layout.batch_dim`** (int, default: 0): Batch dimension index (only used if `auto_detect: false`).
- **`postprocessing.layout.detection_dim`** (int, default: 1): Detection dimension index (only used if `auto_detect: false`).
- **`postprocessing.layout.feature_dim`** (int, default: 2): Feature dimension index (only used if `auto_detect: false`).
- **`postprocessing.layout.bbox_start_idx`** (int, default: 0): Starting index for bbox coordinates (only used if `auto_detect: false`).
- **`postprocessing.layout.bbox_count`** (int, default: 4): Number of bbox coordinates (only used if `auto_detect: false`).
- **`postprocessing.layout.score_idx`** (int, default: 4): Index for confidence score (only used if `auto_detect: false`).
- **`postprocessing.layout.class_idx`** (int, default: 5): Index for class ID (only used if `auto_detect: false`).

**Removed Parameters (use defaults):**
- **`postprocessing.class_score_start_idx`**: Always -1 (use all classes). Not configurable.
- **`postprocessing.class_score_count`**: Always -1 (use all classes). Not configurable.

### Execution Provider Configuration

- **`preferred_provider`** (string, default: "tensorrt"): Execution provider for inference.
  - `"tensorrt"`: TensorRT execution provider (requires CUDA and TensorRT)
  - `"cuda"`: CUDA execution provider (requires CUDA)
  - `"cpu"`: CPU execution provider (always available)
- **`device_id`** (int, default: 0): GPU device ID (for tensorrt/cuda providers).
- **`enable_trt_engine_cache`** (bool, default: false): Enable TensorRT engine caching to disk for faster subsequent startups.
- **`trt_engine_cache_path`** (string, default: "/tmp/deep_ros_ort_trt_cache"): Directory path for TensorRT engine cache.

**Hardcoded Assumptions:**
- **`warmup_tensor_shapes`**: Always enabled. Tensor shapes are always warmed up for optimal performance. Not configurable.

### Batching Configuration

- **`max_batch_size`** (int, default: 3): Number of images required before processing a batch. Should match the number of cameras in your synchronized multi-camera setup.
- **`queue_size`** (int, default: 10): Maximum number of images in the processing queue. Set to 0 for unlimited queue size.

**Hardcoded Assumptions:**
- **Queue overflow policy**: Always `"drop_oldest"`. When queue is full, oldest images are dropped to make room for new ones. Not configurable.
- **Decode failure policy**: Always `"drop"`. Failed image decodes are logged and dropped. Not configurable.
- **Batch timer period**: Hardcoded at 5ms for optimal responsiveness.

### Output Configuration

- **`output_detections_topic`** (string, default: "/detections"): Topic name for publishing Detection2DArray messages.

**Hardcoded Assumptions:**
- **`input_qos_reliability`**: Always `"best_effort"` for image topics. Not configurable.

## Topics

### Input Topics
- **`input_topic`**: MultiImage messages (deep_msgs/MultiImage) containing synchronized compressed images from multiple cameras.

**Limitations:**
- The node **only supports MultiImage messages**. Individual camera topics are not supported.
- Images must be compressed (sensor_msgs/CompressedImage). Raw images are not supported.
- Images are decoded immediately in the subscription callback (not in worker thread).

### Output Topics
- **`output_detections_topic`**: Detection2DArray messages (default: "/detections") containing bounding boxes, scores, and class IDs for each image in the batch.

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

## Node Assumptions and Limitations

### Hardcoded Behaviors (Not Configurable)

The following behaviors are hardcoded for simplicity and cannot be changed:

1. **NMS is always enabled**: Non-maximum suppression cannot be disabled.
2. **Coordinates are always transformed**: Bounding box coordinates are always transformed from preprocessed image space to original image space. The `coordinate_space` parameter is not configurable.
3. **Queue overflow always drops oldest**: When the queue is full, the oldest image is always dropped. Cannot be configured to drop newest or throw.
4. **Decode failures always drop**: Failed image decodes are always dropped with a warning. Cannot be configured to throw.
5. **QoS is always best_effort**: Input subscriptions always use `best_effort` reliability. Cannot be configured to use `reliable`.
6. **Pad value is always 114**: Letterbox padding uses gray value 114 (YOLO standard). Not configurable.
7. **Warmup is always enabled**: Tensor shapes are always warmed up for optimal performance. Not configurable.
8. **Class score indices use defaults**: `class_score_start_idx` and `class_score_count` always use -1 (use all classes). Not configurable.

### Conditional Parameters

Some parameters are only used in specific modes:

- **Multi-output parameters** (`output_boxes_idx`, `output_scores_idx`, `output_classes_idx`): Only used when `postprocessing.use_multi_output: true`. If `use_multi_output: false`, these parameters are ignored.
- **Manual layout parameters**: Only used when `postprocessing.layout.auto_detect: false`. If `auto_detect: true` (default), all manual layout parameters are ignored and layout is automatically detected.

### Limitations

1. **MultiImage input only**: The node only supports MultiImage messages. Individual camera topics are not supported.
2. **Compressed images only**: Only compressed images (sensor_msgs/CompressedImage) are supported. Raw images are not supported.
3. **Image decoding in callback**: Image decoding (`cv::imdecode()`) runs synchronously in the subscription callback. With MultiThreadedExecutor this is less problematic, but can still be a bottleneck if decode time dominates or messages arrive faster than decode throughput.
4. **No dynamic reconfiguration**: Parameters cannot be changed at runtime. Node must be reconfigured to change parameters.
5. **Fail-fast provider selection**: If the specified execution provider is unavailable, the node fails immediately. No automatic fallback to other providers.
6. **Layout auto-detection limitations**: Auto-detection works for common layouts but may fail for unusual tensor shapes. In such cases, set `layout.auto_detect: false` and provide manual layout parameters.
7. **Queue backlog**: If inference throughput is lower than input throughput over time, the queue will steadily drop frames and output rate may appear lower or bursty. This is expected behavior when the system is overloaded.

### Performance Considerations

**Executor Configuration**: The node uses `MultiThreadedExecutor` with 4 threads by default (configured in `main.cpp`). This allows subscription callbacks and timer callbacks to run concurrently, preventing starvation.

**Async Batch Processing**: Batch inference runs asynchronously using `std::async` to prevent blocking the timer callback. Only one batch processes at a time to avoid resource contention.

**Potential Bottlenecks**:
- **Image decoding**: `cv::imdecode()` runs in the subscription callback. If decode time dominates or messages arrive faster than decode throughput, the system may fall behind.
- **Inference throughput**: If inference is slower than input rate, frames will be dropped and publish rate may decrease.
- **GPU/CPU thermal throttling**: Compute performance may degrade over time due to thermal constraints, causing throughput to decrease.

**If publish rate drops over time**, consider:
- Moving decode to a worker thread (off the callback)
- Reducing input rate or image size/quality
- Monitoring GPU/CPU temperatures and throttling
- Using raw images with intra-process zero-copy where possible

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
