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
TensorFlow to ONNX conversion helper script
Supports TF SavedModel, Keras H5, and TFLite formats
"""

import argparse
import os
import sys


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import importlib.util

        # Check for required packages using importlib.util.find_spec
        required_packages = ["tensorflow", "tf2onnx", "onnx"]
        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                raise ImportError(f"Package {package} not found")

        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install tensorflow tf2onnx onnx")
        return False


def convert_savedmodel_to_onnx(model_path, output_path, opset=13):
    """Convert TensorFlow SavedModel to ONNX"""

    # Convert using tf2onnx
    import subprocess

    cmd = [
        "python",
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        model_path,
        "--output",
        output_path,
        "--opset",
        str(opset),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Conversion failed: {result.stderr}")
        sys.exit(1)

    print(f"Successfully converted SavedModel to: {output_path}")


def convert_keras_to_onnx(model_path, output_path, opset=13):
    """Convert Keras H5 model to ONNX"""
    import tensorflow as tf
    import tf2onnx
    import onnx

    # Load Keras model
    model = tf.keras.models.load_model(model_path)

    # Get model input signature
    spec = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype),)

    # Convert to ONNX
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=opset
    )

    # Save ONNX model
    onnx.save(model_proto, output_path)
    print(f"Successfully converted Keras model to: {output_path}")


def convert_tflite_to_onnx(model_path, output_path):
    """Convert TFLite model to ONNX"""

    # TFLite to ONNX is complex, provide guidance
    print("TFLite to ONNX conversion requires intermediate steps:")
    print("1. Convert TFLite back to TensorFlow SavedModel")
    print("2. Then convert SavedModel to ONNX")
    print("\nExample code:")
    print(f"""
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="{model_path}")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a concrete function
@tf.function
def model_func(x):
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# Save as SavedModel first
tf.saved_model.save(model_func, "temp_savedmodel")

# Then convert SavedModel to ONNX using this script
""")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert TensorFlow models to ONNX")
    parser.add_argument(
        "input", help="Input model path (SavedModel dir, .h5, or .tflite)"
    )
    parser.add_argument("output", help="Output ONNX model path")
    parser.add_argument(
        "--opset", type=int, default=13, help="ONNX opset version (default: 13)"
    )

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    # Detect model format
    if os.path.isdir(args.input) and os.path.exists(
        os.path.join(args.input, "saved_model.pb")
    ):
        # TensorFlow SavedModel
        convert_savedmodel_to_onnx(args.input, args.output, args.opset)
    elif args.input.endswith(".h5") or args.input.endswith(".keras"):
        # Keras model
        convert_keras_to_onnx(args.input, args.output, args.opset)
    elif args.input.endswith(".tflite"):
        # TFLite model
        convert_tflite_to_onnx(args.input, args.output)
    else:
        print(f"Unknown model format: {args.input}")
        print("Supported formats: SavedModel directory, .h5, .keras, .tflite")
        sys.exit(1)


if __name__ == "__main__":
    main()
