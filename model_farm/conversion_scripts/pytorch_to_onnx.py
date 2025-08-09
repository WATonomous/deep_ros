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
PyTorch to ONNX conversion helper script
Example script showing common patterns for converting PyTorch models
"""

import argparse
import torch
import torch.onnx
import onnx
import sys


def load_pytorch_model(model_path):
    """
    Load PyTorch model from various formats
    """
    if model_path.endswith(".pt") or model_path.endswith(".pth"):
        # Could be state_dict or full model
        checkpoint = torch.load(model_path, map_location="cpu")

        if isinstance(checkpoint, torch.nn.Module):
            # Full model
            return checkpoint
        elif isinstance(checkpoint, dict):
            # State dict - need model architecture
            print("Error: Found state_dict but need model architecture")
            print("Please provide a model loading function")
            sys.exit(1)
    else:
        print(f"Unknown file format: {model_path}")
        sys.exit(1)


def convert_pytorch_to_onnx(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,
    opset_version=11,
):
    """
    Convert PyTorch model to ONNX with common options
    """
    model.eval()

    # Default dynamic axes for batch dimension
    if dynamic_axes is None:
        dynamic_axes = {
            input_names[0]: {0: "batch_size"},
            output_names[0]: {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Successfully converted to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("input", help="Input PyTorch model path (.pt or .pth)")
    parser.add_argument("output", help="Output ONNX model path")
    parser.add_argument(
        "--input-shape",
        type=str,
        required=True,
        help="Input shape as comma-separated values (e.g., '1,3,224,224')",
    )
    parser.add_argument(
        "--input-names", type=str, default="input", help="Comma-separated input names"
    )
    parser.add_argument(
        "--output-names",
        type=str,
        default="output",
        help="Comma-separated output names",
    )
    parser.add_argument(
        "--opset", type=int, default=11, help="ONNX opset version (default: 11)"
    )
    parser.add_argument(
        "--model-loader",
        type=str,
        help="Python file with load_model() function for loading from state_dict",
    )

    args = parser.parse_args()

    # Parse input shape
    input_shape = [int(x) for x in args.input_shape.split(",")]
    dummy_input = torch.randn(*input_shape)

    # Load model
    if args.model_loader:
        # Import custom loader
        import importlib.util

        spec = importlib.util.spec_from_file_location("model_loader", args.model_loader)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        model = loader.load_model(args.input)
    else:
        model = load_pytorch_model(args.input)

    # Convert
    input_names = args.input_names.split(",")
    output_names = args.output_names.split(",")

    convert_pytorch_to_onnx(
        model,
        dummy_input,
        args.output,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
