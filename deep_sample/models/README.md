# Models Directory

This directory contains ONNX models for the deep_sample package.

## Creating a Simple Model

Since torch and onnx packages are not available in this environment, you can create a simple ONNX model using:

1. **Using PyTorch (if available):**

   ```bash
   python3 scripts/create_simple_model.py
   ```

2. **Using ONNX directly (if available):**

```bash
   python3 scripts/create_dummy_model.py
   ```

1. **Manually place a model:**
   Place any ONNX model file here and update the configuration to point to it.

## Expected Model Format

The sample node expects a model with:
- Input: `[1, 3, 224, 224]` (batch_size, channels, height, width)
- Output: `[1, N]` where N is the number of output features

The model file should be named `simple_model.onnx` or update the configuration accordingly.
