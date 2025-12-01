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
