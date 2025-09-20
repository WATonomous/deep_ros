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
Create a dummy ONNX model for testing using only ONNX (no PyTorch dependency).
This creates a simple linear model that just sums the input features.
"""

import onnx
from onnx import helper, TensorProto
import numpy as np
import os


def create_dummy_model():
    """Create a very simple ONNX model that sums input features"""

    # Define input and output
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    # Create a simple reshape operation followed by a linear transformation
    # Reshape: [1, 3, 224, 224] -> [1, 150528]
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["input", "reshape_shape"],
        outputs=["reshaped"],
        name="reshape",
    )

    # Create weight matrix for simple linear layer
    # We'll create a simple averaging operation
    weights = (
        np.ones((150528, 10), dtype=np.float32) / 150528
    )  # Average and project to 10 dims
    weights_tensor = helper.make_tensor(
        "weights", TensorProto.FLOAT, [150528, 10], weights.flatten()
    )

    # MatMul operation
    matmul_node = helper.make_node(
        "MatMul", inputs=["reshaped", "weights"], outputs=["output"], name="matmul"
    )

    # Reshape shape constant
    reshape_shape_tensor = helper.make_tensor(
        "reshape_shape", TensorProto.INT64, [2], [1, 150528]
    )

    # Create the graph
    graph = helper.make_graph(
        nodes=[reshape_node, matmul_node],
        name="simple_model",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weights_tensor, reshape_shape_tensor],
    )

    # Create the model
    model = helper.make_model(graph, producer_name="deep_sample")
    model.opset_import[0].version = 11

    # Save the model
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "simple_model.onnx"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    onnx.save(model, output_path)
    print(f"Simple model saved to: {output_path}")

    # Verify the model
    try:
        onnx.checker.check_model(model)
        print("Model verification successful!")
    except Exception as e:
        print(f"Model verification failed: {e}")


if __name__ == "__main__":
    create_dummy_model()
