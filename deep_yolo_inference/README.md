# deep_yolo_inference

ROS 2 node for YOLO inference using ONNX Runtime with TensorRT / CUDA / CPU fallback and dynamic batching. The node subscribes to images, batches up to a configurable limit, preprocesses to the YOLO input size, runs inference via the deep_ort plugins, and publishes detections.

## Build
```bash
colcon build --packages-select deep_yolo_inference
source install/setup.bash
```

## Run (example)
```bash
source install/setup.bash
ros2 launch deep_yolo_inference yolo_inference.launch.py
```
# The launch file defaults to `config/yolo_trt.yaml`, which is derived from
# `deep_object_detection/config/object_detection_params.yaml` (model path, batch
# size, thresholds, etc.). The default config now targets the NuScenes bag topics
# (`/CAM_FRONT/image_rect_compressed` with `compressed` transport and
# best-effort QoS), so you can launch against that dataset without overrides.

## TensorRT notes
- The ORT GPU vendor package always downloads the standard GPU tarball (no separate TensorRT artifact upstream). The TensorRT provider loads from your system CUDA/TensorRT libraries; ensure they are installed and visible via `LD_LIBRARY_PATH`. The launch file prepends the vendor lib dir automatically.
- TensorRT engine caching is enabled by default in the GPU backend and stored at `/tmp/deep_ros_ort_trt_cache`. First launch will build; subsequent launches reuse cached engines and start much faster. Override by setting `TRT_ENGINE_CACHE_PATH` in the environment if you prefer a different location.
- The YOLO decoder handles the raw YOLOv8-style output layout (`[N, 84, anchors]` channel-first) and applies objectness * class score with NMS. Use `score_threshold` / `nms_iou_threshold` in the YAML to tune.
- Batching: `batch_size_limit` sets the max batch, but `max_batch_latency_ms` controls how long the node waits for enough images before flushing. Increase latency (e.g., 150–200 ms for ~10 Hz bags) if you want consistent batch >1. Export your ONNX with dynamic batch if you need batch sizes larger than 1.

## Switch providers
- YAML: set `preferred_provider` to `tensorrt` | `cuda` | `cpu` in `config/yolo_trt.yaml`.
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
- Input: `/camera/image_raw` (`sensor_msgs::msg::Image`)
- Output: `/detections` (`deep_msgs::msg::Detection2DArray`)
