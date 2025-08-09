#!/usr/bin/env python3
# Copyright (c) 2025-present WATonomous. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Hugging Face Transformers to ONNX conversion helper
Supports common transformer models (BERT, GPT, ViT, etc.)
"""

import argparse
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import importlib.util

        # Check for required packages using importlib.util.find_spec
        required_packages = ["transformers", "torch", "onnx"]
        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                raise ImportError(f"Package {package} not found")

        # Check for transformers.onnx.export
        if importlib.util.find_spec("transformers.onnx") is None:
            raise ImportError("transformers.onnx not found")

        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers torch onnx")
        return False


def get_model_type(model_name_or_path):
    """Detect model type from config"""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path)
    model_type = config.model_type

    # Map model types to their ONNX configs
    supported_types = {
        "bert": "bert",
        "gpt2": "gpt2",
        "distilbert": "distilbert",
        "roberta": "roberta",
        "vit": "vit",
        "clip": "clip",
        "t5": "t5",
        "bart": "bart",
    }

    if model_type not in supported_types:
        print(f"Warning: Model type '{model_type}' may not be fully supported")
        print(f"Supported types: {', '.join(supported_types.keys())}")

    return model_type


def convert_transformers_to_onnx(
    model_name_or_path, output_path, task="default", opset=14
):
    """Convert Hugging Face model to ONNX using optimum"""
    try:
        # Try using optimum library (recommended)
        print("Using optimum library for conversion...")

        # Export using optimum
        from optimum.exporters.onnx import export as onnx_export
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Export to ONNX
        onnx_export(
            model=model, tokenizer=tokenizer, output=Path(output_path), opset=opset
        )

    except ImportError:
        # Fallback to transformers.onnx
        print("optimum not found, using transformers.onnx...")

        from transformers import AutoModel, AutoTokenizer
        from transformers.onnx import export

        model = AutoModel.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Get model type for validation
        get_model_type(model_name_or_path)

        # Export
        export(
            preprocessor=tokenizer,
            model=model,
            config=model.config,
            opset=opset,
            output=Path(output_path),
        )

    print(f"Successfully converted to: {output_path}")

    # Verify the model
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face Transformers models to ONNX"
    )
    parser.add_argument(
        "input", help="Model name (e.g., 'bert-base-uncased') or local path"
    )
    parser.add_argument("output", help="Output ONNX model path")
    parser.add_argument(
        "--task",
        type=str,
        default="default",
        help="Task type (e.g., 'sequence-classification', 'token-classification')",
    )
    parser.add_argument(
        "--opset", type=int, default=14, help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Apply ONNX optimizations for inference"
    )

    args = parser.parse_args()

    if not check_dependencies():
        print("\nFor best results, also install: pip install optimum[exporters]")
        sys.exit(1)

    convert_transformers_to_onnx(
        args.input, args.output, task=args.task, opset=args.opset
    )

    if args.optimize:
        print("\nOptimizing ONNX model for inference...")
        try:
            from onnxruntime.transformers import optimizer
            from onnxruntime.transformers.fusion_options import FusionOptions

            opt_options = FusionOptions("bert")
            optimizer.optimize_model(
                args.output,
                model_type="bert",
                num_heads=12,
                hidden_size=768,
                optimization_options=opt_options,
            ).save_model_to_file(args.output)
            print("Optimization complete")
        except ImportError:
            print(
                "Install onnxruntime-tools for optimization: pip install onnxruntime-tools"
            )


if __name__ == "__main__":
    main()
