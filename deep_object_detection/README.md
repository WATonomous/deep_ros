# Deep Object Detection Node

A deep learning object detection node for ROS 2. Simply provide any ONNX-compatible object detection model, and the node will automatically detect and adapt to its output format.

## Overview

The `deep_object_detection` package provides:
- Model-agnostic object detection using ONNX models
- Automatic output format detection and adaptation
- Multi-camera support via MultiImage messages (immediate batch processing)
- Configurable preprocessing and postprocessing pipelines
- Plugin-based backend architecture (CPU, CUDA, TensorRT)
- Full ROS 2 lifecycle node support

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                      ROS 2 Topics                                  │
│                                                                                    │
│  ┌──────────────────────────────────┐       ┌──────────────────────────────┐       │
│  │ MultiImage Topic                 │       │ Detection2DArray Topic       │       │
│  │ (synchronized multi-camera)       │       │ (one per image in batch)     │       │
│  │ • KeepLast(1) QoS                │       └──────────────────────────────┘       │
│  │ • Immediate processing           │       ┌──────────────────────────────┐       │
│  └───────────────┬──────────────────┘       │ ImageMarker Topic            │       │
│                  │                          │ (visualization annotations)  │       │
│                  │                          └──────────────────────────────┘       │
└──────────────────┼───────────────────────────────────────────────┬────────────────┘
                   │                                               │
                   ▼                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│          DeepObjectDetectionNode  (rclcpp_lifecycle::LifecycleNode)                │
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                         Processing Pipeline (Per MultiImage Message)         │  │
│  │                                                                              │  │
│  │  MultiImage Message                                                          │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 1. Decode Images (sequential)                                         │  │  │
│  │  │    • Extract all CompressedImage from MultiImage                      │  │  │
│  │  │    • Decode each to cv::Mat                                           │  │  │
│  │  └────────────────────┬───────────────────────────────────────────────────┘  │  │
│  │                       │                                                      │  │
│  │                       ▼                                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 2. ImagePreprocessor (sequential preprocessing)                      │  │  │
│  │  │    • Resize each image (letterbox/resize/crop)                      │  │  │
│  │  │    • Normalize pixel values                                          │  │  │
│  │  │    • Convert color format (RGB/BGR)                                  │  │  │
│  │  └────────────────────┬───────────────────────────────────────────────────┘  │  │
│  │                       │                                                      │  │
│  │                       ▼                                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 3. Pack Batch Tensor                                                 │  │  │
│  │  │    • Stack all preprocessed images into single tensor               │  │  │
│  │  │    • Shape: [batch_size, channels, height, width]                   │  │  │
│  │  └────────────────────┬───────────────────────────────────────────────────┘  │  │
│  │                       │                                                      │  │
│  │                       ▼                                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 4. Backend Plugin (batch inference)                                  │  │  │
│  │  │    • Memory allocation (CPU/GPU)                                    │  │  │
│  │  │    • Single inference call on entire batch                           │  │  │
│  │  │    • Returns batch output tensor                                     │  │  │
│  │  └────────────────────┬───────────────────────────────────────────────────┘  │  │
│  │                       │                                                      │  │
│  │                       ▼                                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 5. GenericPostprocessor (batch postprocessing)                       │  │  │
│  │  │    • Decode batch outputs to detections                               │  │  │
│  │  │    • Apply NMS per image                                              │  │  │
│  │  │    • Transform coordinates (model → image space)                     │  │  │
│  │  └────────────────────┬───────────────────────────────────────────────────┘  │  │
│  │                       │                                                      │  │
│  │                       ▼                                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 6. Publish Results                                                    │  │  │
│  │  │    • One Detection2DArray per image in batch                          │  │  │
│  │  │    • One ImageMarker per image (optional)                            │  │  │
│  │  └──────────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┬─────────────────┘
                                                                   │
                                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           Backend Plugins  (pluginlib)                             │
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │ ClassLoader<DeepBackendPlugin>                                               │  │
│  │                                                                              │  │
│  │  ┌──────────────────────────┐      ┌──────────────────────────┐              │  │
│  │  │ onnxruntime_cpu          │      │ onnxruntime_gpu          │              │  │
│  │  │ (CPU Backend)            │      │ (CUDA / TensorRT)        │              │  │
│  │  └─────────────┬────────────┘      └─────────────┬────────────┘              │  │
│  │                └───────────────┬──────────────────┘                          │  │
│  │                                ▼                                             │  │
│  │  ┌────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ BackendMemoryAllocator                                                 │  │  │
│  │  │ (CPU/GPU memory management)                                            │  │  │
│  │  └────────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ BackendInferenceExecutor                                               │  │  │
│  │  │ (ONNX Runtime batch inference)                                         │  │  │
│  │  └────────────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┬─────────────────┘
                                                                   │
                                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                               Model & Configuration                                │
│                                                                                    │
│  ┌──────────────────────────┐        ┌──────────────────────────────────────────┐  │
│  │ ONNX Model File (.onnx)  │        │ YAML Config                              │  │
│  │                          │        │  • Model.*                              │  │
│  │                          │        │  • Preprocessing.*                      │  │
│  │                          │        │  • Postprocessing.*                     │  │
│  │                          │        │  • Backend.*                            │  │
│  └──────────────────────────┘        └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────┘

Note: Each MultiImage message is processed immediately. All images within a MultiImage
message are processed as a single batch through inference. No queuing or batching of
multiple MultiImage messages occurs.
```

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

2. **With topic remappings**:

   ```bash
   ros2 launch deep_object_detection deep_object_detection.launch.yaml \
     config_file:=/path/to/config.yaml \
     input_topic:=/my_camera/multi_image \
     output_detections_topic:=/my_detections
   ```

3. **With specific model path**:

   ```bash
   ros2 launch deep_object_detection deep_object_detection.launch.yaml \
     config_file:=/path/to/config.yaml \
     preferred_provider:=tensorrt
   ```

4. **Manual lifecycle transitions** (if not using lifecycle manager):

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
     config_file:=/path/to/config.yaml \
     input_topic:=/multi_camera_sync/multi_image_compressed
   ```

   Configure the detection node to subscribe to the MultiImage topic either via:
   - **Remapping** (recommended): `input_topic:=/multi_camera_sync/multi_image_compressed` in launch command
   - **Parameter**: Set `input_topic: "/multi_camera_sync/multi_image_compressed"` in config file

   The node processes all images in each MultiImage message as a batch. The batch size
   is determined by the number of images in the MultiImage message (typically equal to
   the number of synchronized cameras).

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

The node uses a nested parameter structure for configuration. Many parameters have sensible defaults.

### Configuration Example

```yaml
deep_object_detection_node:
  ros__parameters:
    # Required
    model_path: "/path/to/model.onnx"  # Absolute path to ONNX model file
    input_topic: "/multi_camera_sync/multi_image_compressed"

    # Model configuration
    model:
      num_classes: 80
      bbox_format: "cxcywh"  # cxcywh, xyxy, or xywh
    class_names_path: "/path/to/classes.txt"  # Optional: absolute path to class names file

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
      score_activation: "sigmoid"  # sigmoid, softmax, none
      class_score_mode: "all_classes"  # all_classes or single_confidence
      use_multi_output: false
      layout:
        auto_detect: true

    # Backend configuration
    Backend:
      plugin: "onnxruntime_gpu"  # onnxruntime_cpu or onnxruntime_gpu
      execution_provider: "tensorrt"  # tensorrt or cuda (for GPU plugin)
      device_id: 0
      trt_engine_cache_enable: false
      trt_engine_cache_path: "/tmp/deep_ros_ort_trt_cache"

    # Input/Output configuration
    use_compressed_images: true  # true for compressed, false for uncompressed
    output_detections_topic: "/detections"
```

### Backend Plugin Selection

The node uses a plugin-based architecture for backend selection. You must specify which backend plugin to use via the `backend.plugin` parameter.

**Available Plugins:**
- `onnxruntime_cpu` - CPU backend (always available)
- `onnxruntime_gpu` - GPU backend supporting CUDA and TensorRT (requires CUDA)

For GPU plugin, you can specify the execution provider via `backend.execution_provider`:
- `cuda` - CUDA execution provider
- `tensorrt` - TensorRT execution provider (requires TensorRT)

## Expected Model Format

The node automatically detects and adapts to various ONNX model output formats:

- `[batch, detections, features]` - standard format
- `[batch, features, detections]` - transposed format
- `[batch, queries, 4+classes]` - query-based models (e.g., DETR)
- `[batch, channels, anchors]` - anchor-based models
- Any other layout - automatically detected and handled

**Input:** The node expects MultiImage messages (deep_msgs/MultiImage) containing synchronized compressed images from multiple cameras.

**Output:** Detection2DArray messages containing bounding boxes, scores, and class IDs for each image in the batch.

## Parameters

**Important:** All string parameters must be specified in **lowercase**. The node performs direct string comparisons and does not normalize case. For example:
- `model.bbox_format` must be `"cxcywh"`, `"xyxy"`, or `"xywh"` (not `"CXCYWH"`)
- `preprocessing.normalization_type` must be `"imagenet"`, `"scale_0_1"`, `"custom"`, or `"none"`
- `preprocessing.resize_method` must be `"letterbox"`, `"resize"`, `"crop"`, or `"pad"`
- `postprocessing.score_activation` must be `"sigmoid"`, `"softmax"`, or `"none"`
- `postprocessing.class_score_mode` must be `"all_classes"` or `"single_confidence"`
- `backend.plugin` must be `"onnxruntime_cpu"` or `"onnxruntime_gpu"`
- `backend.execution_provider` must be `"cuda"` or `"tensorrt"` (for GPU plugin)

### Required Parameters
- **`model_path`** (string): Absolute path to ONNX model file (e.g., `/workspaces/deep_ros/yolov8m.onnx`).
- **`input_topic`** (string): MultiImage topic name to subscribe to.

### Key Parameters
- **`class_names_path`** (string, optional): Absolute path to text file with class names, one per line (e.g., `/workspaces/deep_ros/deep_object_detection/config/coco_classes.txt`). If not provided, class IDs will be used in output messages.
- **`model.num_classes`** (int, default: 80): Number of detection classes.
- **`model.bbox_format`** (string, default: "cxcywh"): Bounding box format (cxcywh, xyxy, or xywh).
- **`preprocessing.input_width/input_height`** (int, default: 640): Model input image dimensions.
- **`preprocessing.normalization_type`** (string, default: "scale_0_1"): Normalization method.
- **`preprocessing.resize_method`** (string, default: "letterbox"): Image resizing method.
- **`postprocessing.score_threshold`** (float, default: 0.25): Minimum confidence score.
- **`postprocessing.nms_iou_threshold`** (float, default: 0.45): IoU threshold for NMS.
- **`backend.plugin`** (string, required): Backend plugin name (`"onnxruntime_cpu"` or `"onnxruntime_gpu"`).
- **`backend.execution_provider`** (string, default: "tensorrt"): Execution provider for GPU plugin (`"cuda"` or `"tensorrt"`). Only used with `onnxruntime_gpu` plugin.
- **`backend.device_id`** (int, default: 0): GPU device ID (for CUDA/TensorRT).
- **`backend.trt_engine_cache_enable`** (bool, default: false): Enable TensorRT engine caching.
- **`backend.trt_engine_cache_path`** (string, default: "/tmp/deep_ros_ort_trt_cache"): TensorRT engine cache directory.

See `config/generic_model_params.yaml` for a complete parameter reference.

## Topics

Topic names can be configured either via parameters in the config file or via remappings in the launch file. Remappings take precedence over parameter values.

### Input Topics
- **`input_topic`**: MultiImage messages (deep_msgs/MultiImage) containing synchronized compressed images from multiple cameras.

**Note:** The node only supports MultiImage messages. Individual camera topics are not supported.

**Configuration:**
- Via parameter: Set `input_topic` in the config file
- Via remapping: Use `input_topic:=/your/topic/name` in the launch command or add remapping in launch file

### Output Topics
- **`output_detections_topic`**: Detection2DArray messages (default: "/detections") containing bounding boxes, scores, and class IDs for each image in the batch.

**Configuration:**
- Via parameter: Set `output_detections_topic` in the config file
- Via remapping: Use `output_detections_topic:=/your/topic/name` in the launch command or add remapping in launch file

**Example launch file with remappings:**

```yaml
- node:
    pkg: "deep_object_detection"
    exec: "deep_object_detection_node"
    name: "deep_object_detection_node"
    remap:
      - from: "/multi_camera_sync/multi_image_compressed"
        to: "/my_camera/multi_image"
      - from: "/detections"
        to: "/my_detections"
    param:
      - from: "$(var config_file)"
```

## Limitations

1. **MultiImage input only**: The node only supports MultiImage messages. Individual camera topics are not supported.
2. **MultiImage input format**: The node processes MultiImage messages immediately upon receipt. All images within a MultiImage message are processed as a single batch. No queuing or accumulation of multiple MultiImage messages occurs.
3. **No dynamic reconfiguration**: Parameters cannot be changed at runtime. Node must be reconfigured to change parameters.

## Testing

The package includes both unit tests (C++) and launch tests (Python) for comprehensive testing.

### Unit Tests (C++)

Fast unit tests using the `deep_test` framework that verify node construction, parameter handling, and lifecycle state management without requiring model files or GPU access.

**Run unit tests:**

```bash
# Build with tests enabled
colcon build --packages-select deep_object_detection --cmake-args -DBUILD_TESTING=ON

# Run tests
source install/setup.bash
colcon test --packages-select deep_object_detection

# View results
colcon test-result --verbose
```

**Run specific test:**

```bash
./build/deep_object_detection/test_deep_object_detection_node "[node][construction]"
```

### Launch Tests (Python)

Integration tests that launch the full node with different backends (CPU, CUDA, TensorRT) and verify end-to-end functionality including model loading, inference, and detection output.

**Note:** Launch tests are **disabled by default** to keep test runs fast. They require model loading which is slow (~30-60 seconds per test).

**Run launch tests (opt-in, requires model file):**

```bash
# Enable launch tests explicitly
export ENABLE_LAUNCH_TESTS=1
# Build and run
colcon build --packages-select deep_object_detection --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
colcon test --packages-select deep_object_detection

# View results
colcon test-result --verbose
```

**Available launch tests:**
- `test_deep_object_detection_cpu_backend.py` - CPU backend test
- `test_deep_object_detection_gpu_backend.py` - CUDA backend test (requires GPU)
- `test_deep_object_detection_tensorrt_backend.py` - TensorRT backend test (requires GPU + TensorRT)

Launch tests are disabled by default and must be explicitly enabled with `ENABLE_LAUNCH_TESTS=1`.

**Test Requirements:**
- Model file: `/workspaces/deep_ros/yolov8m.onnx` (for launch tests)
- Class names file: `/workspaces/deep_ros/deep_object_detection/config/coco_classes.txt`
- GPU access: Required for GPU and TensorRT backend tests (local only)
