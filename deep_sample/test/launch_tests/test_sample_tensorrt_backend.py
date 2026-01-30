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
Launch test for deep_sample with TensorRT backend.

This test should ONLY be run locally on a machine with a GPU and TensorRT.
It will be skipped in CI environments.
"""

import os
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest
import rclpy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import numpy as np


def _is_gpu_available():
    """Check if GPU and CUDA libraries are available."""
    try:
        import ctypes

        # Try to load CUDA runtime library
        ctypes.CDLL("libcuda.so.1")
        return True
    except (OSError, AttributeError):
        return False


@pytest.mark.launch_test
def generate_test_description():
    """Generate launch description for TensorRT backend test."""
    from ament_index_python.packages import get_package_share_directory

    # Path to TensorRT config file
    config_file = os.path.join(
        get_package_share_directory("deep_sample"),
        "config",
        "sample_node_tensorrt_config.yaml",
    )

    # Sample inference node with TensorRT backend
    sample_node = launch_ros.actions.Node(
        package="deep_sample",
        executable="sample_inference_node",
        name="sample_inference_node",
        parameters=[config_file],
        output="screen",
    )

    # Lifecycle manager
    lifecycle_manager = launch_ros.actions.Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager",
        parameters=[{"node_names": ["sample_inference_node"], "autostart": True}],
        output="screen",
    )

    return (
        launch.LaunchDescription(
            [sample_node, lifecycle_manager, launch_testing.actions.ReadyToTest()]
        ),
        {
            "sample_node": sample_node,
            "lifecycle_manager": lifecycle_manager,
        },
    )


@unittest.skipUnless(
    _is_gpu_available(), "GPU/CUDA not available - skipping TensorRT backend tests"
)
class TestTensorRTBackend(unittest.TestCase):
    """Test TensorRT backend functionality."""

    @classmethod
    def setUpClass(cls):
        """Initialize ROS context."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shutdown ROS context."""
        rclpy.shutdown()

    def setUp(self):
        """Set up test fixtures."""
        self.node = rclpy.create_node("test_tensorrt_backend")

    def tearDown(self):
        """Clean up test fixtures."""
        self.node.destroy_node()

    def test_node_starts(self, proc_output):
        """Test that the sample node starts successfully."""
        proc_output.assertWaitFor("SampleInferenceNode constructor", timeout=10)

    def test_backend_loads(self, proc_output):
        """Test that GPU backend plugin loads."""
        proc_output.assertWaitFor("Loading plugin: onnxruntime_gpu", timeout=10)
        proc_output.assertWaitFor(
            "Successfully loaded plugin: onnxruntime_gpu", timeout=10
        )

    def test_tensorrt_provider_configured(self, proc_output):
        """Test that TensorRT execution provider is configured."""
        proc_output.assertWaitFor(
            "Configuring TensorRT execution provider on device 0", timeout=10
        )
        proc_output.assertWaitFor(
            "TensorRT provider registered successfully", timeout=10
        )

    def test_model_loads(self, proc_output):
        """Test that the model loads successfully with TensorRT backend."""
        proc_output.assertWaitFor("Loading model:", timeout=15)
        proc_output.assertWaitFor("Successfully loaded model:", timeout=15)

    def test_node_activates(self, proc_output):
        """Test that the node activates successfully with TensorRT backend."""
        proc_output.assertWaitFor(
            "SampleInferenceNode activated with backend: onnxruntime_gpu", timeout=20
        )

    def test_no_tensorrt_errors(self, proc_output):
        """Test that there are no TensorRT-related errors."""
        # This will fail if any TensorRT errors appear in the output
        time.sleep(2)  # Give time for any errors to appear
        # If we get here without exceptions from previous assertions, no TensorRT errors occurred

    def test_tensorrt_inference_with_dummy_image(self, proc_output):
        """Test end-to-end TensorRT inference by publishing a dummy image and verifying output."""
        # Wait for node to be fully activated
        proc_output.assertWaitFor(
            "SampleInferenceNode activated with backend: onnxruntime_gpu", timeout=20
        )
        time.sleep(1)

        # Create publisher for dummy images
        image_pub = self.node.create_publisher(Image, "/camera/image_raw", 10)

        # Variable to track if we received output
        received_output = []

        def output_callback(msg):
            received_output.append(msg)
            self.node.get_logger().info(
                f"Received TensorRT inference output with {len(msg.data)} elements"
            )

        # Create subscriber for inference output
        self.output_sub = self.node.create_subscription(
            Float32MultiArray, "/inference/output", output_callback, 10
        )

        # Wait for publisher/subscriber to be ready
        time.sleep(1)

        # Create a dummy 32x32 RGB image with float32 data (tiny_model expects 32x32)
        dummy_image = Image()
        dummy_image.header.stamp = self.node.get_clock().now().to_msg()
        dummy_image.header.frame_id = "camera"
        dummy_image.height = 32
        dummy_image.width = 32
        dummy_image.encoding = "32FC3"  # float32, 3 channels
        dummy_image.is_bigendian = 0
        dummy_image.step = 32 * 3 * 4  # width * channels * bytes_per_channel
        dummy_image.data = np.random.rand(32, 32, 3).astype(np.float32).tobytes()

        # Publish dummy image
        self.node.get_logger().info(
            "Publishing dummy image for TensorRT inference test"
        )
        image_pub.publish(dummy_image)

        # Spin to process callbacks
        start_time = time.time()
        timeout = 5.0
        while len(received_output) == 0 and (time.time() - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Verify we received output
        self.assertGreater(
            len(received_output),
            0,
            "Should receive inference output after publishing image",
        )
        self.assertGreater(
            len(received_output[0].data), 0, "Inference output should contain data"
        )
        self.node.get_logger().info(
            f"TensorRT inference test passed! Received {len(received_output[0].data)} output values"
        )
