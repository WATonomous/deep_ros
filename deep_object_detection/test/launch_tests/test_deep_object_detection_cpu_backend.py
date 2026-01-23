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

"""Launch test for deep_object_detection with CPU backend (cpu)."""

import os
import time
import unittest

import cv2
import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import numpy as np
import pytest
import rclpy
from deep_msgs.msg import MultiImage
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from vision_msgs.msg import Detection2DArray


@pytest.mark.launch_test
def generate_test_description():
    """Generate launch description for CPU backend test."""
    from ament_index_python.packages import get_package_share_directory

    # Path to base config file
    config_file = os.path.join(
        get_package_share_directory("deep_object_detection"),
        "config",
        "generic_model_params.yaml",
    )

    # Deep object detection node with CPU backend
    detection_node = launch_ros.actions.Node(
        package="deep_object_detection",
        executable="deep_object_detection_node",
        name="deep_object_detection_node",
        parameters=[
            config_file,
            {"preferred_provider": "cpu"},
            {"enable_trt_engine_cache": False},
        ],
        output="screen",
    )

    # Lifecycle manager
    lifecycle_manager = launch_ros.actions.Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager",
        parameters=[{"node_names": ["deep_object_detection_node"], "autostart": True}],
        output="screen",
    )

    return (
        launch.LaunchDescription(
            [detection_node, lifecycle_manager, launch_testing.actions.ReadyToTest()]
        ),
        {
            "detection_node": detection_node,
            "lifecycle_manager": lifecycle_manager,
        },
    )


class TestCPUBackend(unittest.TestCase):
    """Test CPU backend functionality."""

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
        self.node = rclpy.create_node("test_cpu_backend")

    def tearDown(self):
        """Clean up test fixtures."""
        self.node.destroy_node()

    def test_node_starts(self, proc_output):
        """Test that the detection node starts successfully."""
        proc_output.assertWaitFor("Deep object detection node created", timeout=10)

    def test_backend_loads(self, proc_output):
        """Test that CPU backend loads."""
        proc_output.assertWaitFor("Initialized backend using provider: cpu", timeout=15)

    def test_model_loads(self, proc_output):
        """Test that the model loads successfully."""
        # Model loading happens during backend initialization
        # Check for configuration completion which indicates model is loaded
        proc_output.assertWaitFor("Deep object detection node configured", timeout=20)

    def test_node_activates(self, proc_output):
        """Test that the node activates successfully."""
        proc_output.assertWaitFor("Deep object detection node activated", timeout=20)

    def test_detection_with_dummy_multiimage(self, proc_output):
        """Test end-to-end detection by publishing a dummy MultiImage and verifying output."""
        # Wait for node to be fully activated
        proc_output.assertWaitFor("Deep object detection node activated", timeout=20)
        time.sleep(1)

        # Create publisher for MultiImage messages
        multi_image_pub = self.node.create_publisher(
            MultiImage, "/multi_camera_sync/multi_image_compressed", 10
        )

        # Variable to track if we received detections
        received_detections = []

        def detection_callback(msg):
            received_detections.append(msg)
            self.node.get_logger().info(
                f"Received detection output with {len(msg.detections)} detections"
            )

        # Create subscriber for detection output with matching QoS
        from rclpy.qos import SensorDataQoS

        qos_profile = SensorDataQoS()
        self.detection_sub = self.node.create_subscription(
            Detection2DArray, "/detections", detection_callback, qos_profile
        )

        # Wait for publisher/subscriber to be ready
        time.sleep(2)

        # Create a dummy compressed image (640x640 RGB JPEG)
        # This matches the expected input size from the config
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_image[:, :] = [128, 128, 128]  # Gray image
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded_image = cv2.imencode(".jpg", dummy_image, encode_param)

        # Create MultiImage message with 1 image (for testing)
        # Note: max_batch_size is 6, but node will process when batch timer fires
        multi_image_msg = MultiImage()
        multi_image_msg.header = Header()
        multi_image_msg.header.stamp = self.node.get_clock().now().to_msg()
        multi_image_msg.header.frame_id = "camera"

        # Create compressed image message
        compressed_img = CompressedImage()
        compressed_img.header = multi_image_msg.header
        compressed_img.format = "jpeg"
        compressed_img.data = encoded_image.tobytes()

        multi_image_msg.images = [compressed_img]

        # Publish MultiImage message multiple times to fill batch
        # (max_batch_size=6, but node processes when timer fires with >= max_batch_size)
        self.node.get_logger().info(
            "Publishing dummy MultiImage for CPU detection test"
        )
        for _ in range(6):
            multi_image_pub.publish(multi_image_msg)
            time.sleep(0.01)  # Small delay between publishes

        # Spin to process callbacks and wait for batch processing
        start_time = time.time()
        timeout = 15.0
        while len(received_detections) == 0 and (time.time() - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Verify we received detections (even if empty, the message should be published)
        self.assertGreater(
            len(received_detections),
            0,
            "Should receive detection output after publishing MultiImage",
        )
        self.node.get_logger().info(
            f"CPU detection test passed! Received {len(received_detections[0].detections)} detections"
        )
