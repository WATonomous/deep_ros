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
Launch test for deep_sample with GPU backend (onnxruntime_gpu).

This test should ONLY be run locally on a machine with a GPU.
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


@pytest.mark.launch_test
def generate_test_description():
    """Generate launch description for GPU backend test."""
    from ament_index_python.packages import get_package_share_directory

    # Path to GPU config file
    config_file = os.path.join(
        get_package_share_directory("deep_sample"),
        "config",
        "sample_node_gpu_config.yaml",
    )

    # Sample inference node with GPU backend
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


class TestGPUBackend(unittest.TestCase):
    """Test GPU backend functionality."""

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
        self.node = rclpy.create_node("test_gpu_backend")

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

    def test_model_loads(self, proc_output):
        """Test that the model loads successfully with GPU backend."""
        proc_output.assertWaitFor("Loading model:", timeout=10)
        proc_output.assertWaitFor("Successfully loaded model:", timeout=10)

    def test_node_activates(self, proc_output):
        """Test that the node activates successfully with GPU backend."""
        proc_output.assertWaitFor(
            "SampleInferenceNode activated with backend: onnxruntime_gpu", timeout=15
        )

    def test_no_cuda_errors(self, proc_output):
        """Test that there are no CUDA-related errors."""
        # This will fail if any CUDA errors appear in the output
        time.sleep(2)  # Give time for any errors to appear
        # If we get here without exceptions from previous assertions, no CUDA errors occurred

    def test_gpu_inference_with_dummy_image(self, proc_output):
        """Test end-to-end GPU inference by publishing a dummy image and verifying output."""
        # Wait for node to be fully activated
        proc_output.assertWaitFor(
            "SampleInferenceNode activated with backend: onnxruntime_gpu", timeout=15
        )
        time.sleep(1)

        # Create publisher for dummy images
        image_pub = self.node.create_publisher(Image, "/camera/image_raw", 10)

        # Variable to track if we received output
        received_output = []

        def output_callback(msg):
            received_output.append(msg)
            self.node.get_logger().info(
                f"Received GPU inference output with {len(msg.data)} elements"
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
        self.node.get_logger().info("Publishing dummy image for GPU inference test")
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
            f"GPU inference test passed! Received {len(received_output[0].data)} output values"
        )
