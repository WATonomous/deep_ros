# Deep Object Detection Node

ROS 2 node for deep learning object detection using ONNX-compatible models.

This node provides model-agnostic object detection for ROS 2, supporting any ONNX-compatible detection model with configurable preprocessing, postprocessing, and multi-camera batch processing. All processing is lifecycle-managed for clean startup/shutdown.

## Architecture

```
┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
│ MultiImage / MultiImageCompressed    │ ─────► │ DeepObjectDetectionNode              │
│ (multi-camera)                       │        │ (lifecycle node)                     │
└──────────────────────────────────────┘        │  • Decode & preprocess               │
                                                │  • Batch inference                   │
                                                │  • Postprocess & NMS                 │
                                                └───────────────┬──────────────────────┘
                                                                │ publishes
                                                                ▼
┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
│ Backend Plugins                      │ ◄───── │ Detection2DArray                     │
│  • onnxruntime_cpu                   │  uses  │ ImageMarker (optional)               │
│  • onnxruntime_gpu (CUDA/TRT)        │        └──────────────────────────────────────┘
└───────────────┬──────────────────────┘
                │ loads
                ▼
┌──────────────────────────────────────┐
│ ONNX Model + YAML Config             │
└──────────────────────────────────────┘


```

## Running

```bash
ros2 launch deep_object_detection deep_object_detection.launch.yaml \
  config_file:=/path/to/config.yaml
```

Or run directly:

```bash
ros2 run deep_object_detection deep_object_detection_node \
  --ros-args \
  --params-file /path/to/config.yaml
```

## Parameters

### Required

- **`model_path`** (string): Absolute path to ONNX model file (e.g., `/workspaces/deep_ros/yolov8m.onnx`)
- **`input_topic`** (string): MultiImage/MultiImageCompressed topic name to subscribe to

### Model Configuration

- **`Model.num_classes`** (int, default: 80): Number of detection classes
- **`Model.bbox_format`** (string, default: "cxcywh"): Bounding box format (`cxcywh`, `xyxy`, or `xywh`)
- **`Model.output_shape`** (array, optional): Expected model output shape `[batch, detections, features]` (e.g., `[1, 8400, 84]`)
- **`class_names_path`** (string, optional): Absolute path to text file with class names, one per line (e.g., `/workspaces/deep_ros/deep_object_detection/config/coco_classes.txt`)

### Preprocessing

- **`Preprocessing.input_width`** (int, default: 640): Model input image width
- **`Preprocessing.input_height`** (int, default: 640): Model input image height
- **`Preprocessing.normalization_type`** (string, default: "scale_0_1"): Normalization method (`scale_0_1`, `imagenet`, `custom`, `none`)
- **`Preprocessing.resize_method`** (string, default: "letterbox"): Image resizing method (`letterbox`, `resize`, `crop`, `pad`)
- **`Preprocessing.color_format`** (string, default: "rgb"): Color format (`rgb` or `bgr`)
- **`Preprocessing.mean`** (array, default: [0.0, 0.0, 0.0]): Mean values for custom normalization
- **`Preprocessing.std`** (array, default: [1.0, 1.0, 1.0]): Standard deviation values for custom normalization
- **`Preprocessing.pad_value`** (int, default: 114): Padding value for letterbox resizing

### Postprocessing

- **`Postprocessing.score_threshold`** (float, default: 0.65): Minimum confidence score
- **`Postprocessing.nms_iou_threshold`** (float, default: 0.45): IoU threshold for NMS
- **`Postprocessing.score_activation`** (string, default: "sigmoid"): Score activation (`sigmoid`, `softmax`, `none`)
- **`Postprocessing.class_score_mode`** (string, default: "all_classes"): Class score mode (`all_classes` or `single_confidence`)
- **`Postprocessing.enable_nms`** (bool, default: true): Enable non-maximum suppression
- **`Postprocessing.class_score_start_idx`** (int, default: -1): Start index for class scores (-1 for auto)
- **`Postprocessing.class_score_count`** (int, default: -1): Number of class scores (-1 for auto)

#### Postprocessing Layout

- **`Postprocessing.layout.batch_dim`** (int, default: 0): Batch dimension index
- **`Postprocessing.layout.detection_dim`** (int, default: 1): Detection dimension index
- **`Postprocessing.layout.feature_dim`** (int, default: 2): Feature dimension index
- **`Postprocessing.layout.bbox_start_idx`** (int, default: 0): Bounding box start index
- **`Postprocessing.layout.bbox_count`** (int, default: 4): Number of bbox coordinates
- **`Postprocessing.layout.score_idx`** (int, default: 4): Score index
- **`Postprocessing.layout.class_idx`** (int, default: 5): Class index

### Backend

- **`Backend.plugin`** (string, required): Backend plugin name (`onnxruntime_cpu` or `onnxruntime_gpu`)
- **`Backend.execution_provider`** (string, default: "tensorrt"): Execution provider for GPU plugin (`cuda` or `tensorrt`)
- **`Backend.device_id`** (int, default: 0): GPU device ID (for CUDA/TensorRT)
- **`Backend.trt_engine_cache_enable`** (bool, default: true): Enable TensorRT engine caching
- **`Backend.trt_engine_cache_path`** (string, default: "/tmp/deep_ros_ort_trt_cache"): TensorRT engine cache directory

### Input/Output

- **`use_compressed_images`** (bool, default: true): Use compressed images (MultiImageCompressed) vs uncompressed (MultiImage)
- **`output_detections_topic`** (string, default: "/detections"): Output detections topic name

## Topics

### Key Topics

| Topic | Type | Description |
|-------|------|-------------|
| `input_topic` | `deep_msgs/MultiImage` or `deep_msgs/MultiImageCompressed` | Synchronized multi-camera input (compressed or uncompressed) |
| `output_detections_topic` | `vision_msgs/Detection2DArray` | Detection results (one per image in batch, default: `/detections`) |
| `/image_annotations` | `visualization_msgs/ImageMarker` | Visualization annotations with bounding boxes (optional) |

**Note:** The node only supports MultiImage/MultiImageCompressed messages. Individual camera topics are not supported.

### Key Services

| Service | Type | Description |
|---------|------|-------------|
| `/<node_name>/configure` | `lifecycle_msgs/srv/ChangeState` | Configure the lifecycle node |
| `/<node_name>/activate` | `lifecycle_msgs/srv/ChangeState` | Activate the lifecycle node |
| `/<node_name>/deactivate` | `lifecycle_msgs/srv/ChangeState` | Deactivate the lifecycle node |
| `/<node_name>/cleanup` | `lifecycle_msgs/srv/ChangeState` | Cleanup the lifecycle node |
| `/<node_name>/shutdown` | `lifecycle_msgs/srv/ChangeState` | Shutdown the lifecycle node |
| `/<node_name>/get_state` | `lifecycle_msgs/srv/GetState` | Get current lifecycle state |
