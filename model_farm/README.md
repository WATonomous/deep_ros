# Model Farm - Modular ML Project Infrastructure

Sandbox environment for training and converting ML models with isolated dependencies.

## Philosophy

- Each project is self-contained with its own Docker environment
- **Conversion is left to implementers** - we provide examples but don't enforce methods
- Projects can use any framework version without conflicts
- Infrastructure scripts in bash for speed and simplicity
- Specialized tasks (like quantization) in Python

## Structure

```
model_farm/
├── farm                    # Bash script for project management
├── projects/              # Individual ML projects
│   └── example_pytorch/   # Example showing PyTorch->ONNX conversion
├── examples/              # Conversion examples for different frameworks
│   ├── pytorch_to_onnx.py
│   ├── tf_to_onnx.py
│   └── huggingface_to_onnx.py
└── templates/             # Project templates
```

## Conversion Approach

**We don't dictate how to convert models.** Each project knows best how to export their specific model to ONNX. We provide:

1. Example conversion scripts for common frameworks
2. Docker environments with necessary dependencies
3. Infrastructure to run conversions in isolated environments

Projects are responsible for:
- Choosing conversion method
- Handling model-specific requirements (input shapes, preprocessing)
- Ensuring ONNX output is valid
