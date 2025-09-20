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
Create a simple ONNX model for testing the deep_sample package.
This creates a minimal CNN that takes a 224x224x3 image and outputs 10 classification scores.
"""

import torch
import torch.nn as nn
import torch.onnx
import os


class SimpleCNN(nn.Module):
    """Simple CNN for testing purposes"""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_simple_model():
    """Create and export a simple ONNX model"""

    # Create model
    model = SimpleCNN()
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "simple_cnn.onnx"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Simple CNN model exported to: {output_path}")

    # Test the model with ONNX Runtime
    try:
        import onnxruntime as ort

        # Load and test the model
        ort_session = ort.InferenceSession(output_path)

        # Test with dummy input
        input_data = dummy_input.numpy()
        outputs = ort_session.run(None, {"input": input_data})

        print("Model test successful!")
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {outputs[0].shape}")
        print(f"Output range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")

    except ImportError:
        print("ONNX Runtime not available for testing, but model should work fine")
    except Exception as e:
        print(f"Model test failed: {e}")


if __name__ == "__main__":
    create_simple_model()
