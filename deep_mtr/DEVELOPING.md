# Deep MTR Development Boundary

This package owns the future model-specific Motion Transformer implementation. The current node is
only lifecycle and message-contract scaffolding.

The inference owner must implement these seams in `deep_mtr`:

1. Maintain per-track history from repeated `MtrScene` messages.
2. Pack the verified MTR ONNX input contract.
3. Call `DeepNodeBase::run_inference` through `onnxruntime_gpu`.
4. Decode outputs into `MtrPredictionArray`.
5. Add real-model contract, correctness, and GPU tests.

The eventual deployment configuration is expected to select:

```yaml
Backend:
  plugin: "onnxruntime_gpu"
  execution_provider: "tensorrt"
```

The backend must execute an ONNX model through ONNX Runtime's TensorRT execution provider. Do not
load serialized TensorRT `.engine` files directly: that bypasses the `DeepNodeBase` backend contract
and introduces hardware- and runtime-specific artifacts into the ROS interface layer.

WATO-specific lanelet adaptation, request correlation, fallback prediction, and world-model output
remain the responsibility of the consuming bridge, not `deep_mtr` or `deep_msgs`.
