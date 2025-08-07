# ONNX Conversion Scripts

Helper scripts for converting models from various frameworks to ONNX format.

## Available Scripts

### pytorch_to_onnx.py
Converts PyTorch models (.pt, .pth) to ONNX.

```bash
python pytorch_to_onnx.py model.pt model.onnx --input-shape 1,3,224,224
```

Features:
- Automatic format detection
- Dynamic batch size support
- Custom input/output naming
- Model validation

### tensorflow_to_onnx.py
Converts TensorFlow models (SavedModel, Keras .h5, TFLite) to ONNX.

```bash
python tensorflow_to_onnx.py saved_model_dir/ model.onnx
python tensorflow_to_onnx.py model.h5 model.onnx
```

Supported formats:
- TensorFlow SavedModel
- Keras H5/Keras files
- TFLite (with guidance)

### huggingface_to_onnx.py
Converts Hugging Face Transformers models to ONNX.

```bash
python huggingface_to_onnx.py bert-base-uncased bert.onnx
python huggingface_to_onnx.py ./local_model_dir model.onnx --optimize
```

Features:
- Automatic model type detection
- Task-specific exports
- Optional optimization for inference
- Support for BERT, GPT-2, ViT, CLIP, etc.

## Usage in Projects

These scripts are automatically mounted when using the farm tool:

```bash
# From model_farm directory
./farm run my_project python /workspace/conversion_scripts/pytorch_to_onnx.py model.pt model.onnx --input-shape 1,3,224,224
```

## Creating Custom Converters

For models with specific requirements, create custom conversion scripts in your project directory. Use these scripts as examples of common patterns.