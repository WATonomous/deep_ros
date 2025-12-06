# deep_yolo_inference

ROS 2 node for YOLO inference using ONNX Runtime with TensorRT / CUDA / CPU fallback and dynamic batching. The node subscribes to one or more camera topics, batches up to a configurable limit, preprocesses to the YOLO input size, runs inference via the deep_ort plugins, and publishes detections as either `deep_msgs` or `vision_msgs` messages depending on what is available in the workspace.

## Features
- **Execution providers**: prefers TensorRT (engine caching + automatic build logging), falls back to CUDA then CPU if necessary.
- **Fixed batching**: always processes batches of 3 images (matching the three NuScenes front cameras) for consistent TensorRT performance.
- **Multi-camera ingest**: list `camera_topics` to feed synchronized compressed topics; otherwise subscribe to a single raw/compressed stream with image_transport.
- **Automatic message aliasing**: publishes `deep_msgs::Detection2D(Array)` when the package is present; otherwise uses the upstream `vision_msgs` API. No code changes required on downstream consumers beyond selecting the right dependency.
- **Warmup cache**: optional tensor-shape warmup primes TensorRT/CUDA kernels for each batch size before real traffic arrives.

## Build
```bash
colcon build --packages-select deep_yolo_inference
source install/setup.bash
```

## Run (example)
```bash
source install/setup.bash
ros2 launch deep_yolo_inference yolo_inference.launch.py \
  config_file:=install/deep_yolo_inference/share/deep_yolo_inference/config/object_detection_params.yaml
```

The sample `object_detection_params.yaml` configures both `object_detection_node`
and `yolo_inference_node`. Edit camera topics, batching, providers, and scores in
one place, re-launch, and both pipelines stay in sync. By default the YAML lists
the NuScenes front/left/right compressed topics and sets `batch_size_limit: 3`
so the YOLO node processes a batch containing all three frames.

## Key parameters (see YAML for defaults)
| Parameter | Description |
| --- | --- |
| `model_path` | Absolute path to the exported YOLO ONNX file (required).
| `preferred_provider` | `tensorrt`, `cuda`, or `cpu`. The node auto-falls back if init fails.
| `device_id` | GPU ID to pass to the deep ORT GPU backend.
| `camera_topics` | Optional list of compressed topics for multi-camera batching. Leave empty for single input.
| `input_image_topic` / `input_transport` | Single stream input topic + desired image_transport.
| `batch_size_limit` | Fixed at 3. Any other value is ignored to keep TensorRT engine usage predictable.
| `score_threshold` / `nms_iou_threshold` | Detection filtering parameters.
| `warmup_tensor_shapes` | When true, runs dummy inferences for batch sizes 1..N to build TensorRT engines up front.

See `config/object_detection_params.yaml` for additional QoS and preprocessing knobs.

## TensorRT notes
- The ORT GPU vendor package always downloads the standard GPU tarball (no separate TensorRT artifact upstream). The TensorRT provider loads from your system CUDA/TensorRT libraries; ensure they are installed and visible via `LD_LIBRARY_PATH`. The launch file prepends the vendor lib dir automatically.
- The deep ORT GPU backend logs when a TensorRT engine build starts and finishes (per batch size). Expect a one-time build per unique batch dimension; subsequent runs reuse the cached engine.
- Default engine cache lives under `/tmp/deep_ros_ort_trt_cache`. Override by exporting `TRT_ENGINE_CACHE_PATH` before launching if you want a persistent location.
- The YOLO decoder handles the raw YOLOv8-style output layout (`[N, channels, anchors]` channel-first) and applies objectness * class score with NMS. Use `score_threshold` / `nms_iou_threshold` to tune.
- Batching reminders: The node requires exactly three queued frames before issuing inference. Ensure your camera topics publish in sync; otherwise frames will queue until the trio is available.

## Switch providers
- YAML: set `preferred_provider` to `tensorrt` | `cuda` | `cpu` in `config/object_detection_params.yaml`.
- CLI override example:
  ```bash
  ros2 run deep_yolo_inference yolo_inference_node \
    --ros-args --params-file config/yolo_trt.yaml \
    -p preferred_provider:=cuda
  ```

## Tests
```bash
colcon test --packages-select deep_yolo_inference
colcon test-result --verbose
```

## Topics
- Input: `/camera/image_raw` (`sensor_msgs::msg::Image`) or the list in `camera_topics`
- Output: `/detections` (`deep_msgs::msg::Detection2DArray` if available, otherwise `vision_msgs::msg::Detection2DArray`)

Downstream packages that expect one specific message type should add either
`deep_msgs` or `vision_msgs` to their manifest accordingly.
