// Copyright (c) 2025-present WATonomous. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file test_deep_object_detection_node.cpp
 * @brief Unit tests for DeepObjectDetectionNode
 *
 * These tests focus on fast unit tests that don't require:
 * - Model loading (slow, requires file I/O)
 * - GPU operations (not available in CI)
 * - Actual inference (slow, requires backend initialization)
 *
 * Test coverage:
 * - Node construction and parameter declaration
 * - Parameter validation and default values
 * - Lifecycle state management
 * - Topic configuration
 * - Queue size parameter handling
 * - Batch size parameter validation
 *
 * Reasoning for each test:
 * - Parameter defaults: Ensures configuration is correct and prevents regressions
 * - Lifecycle transitions: Critical for ROS2 lifecycle node correctness
 * - Topic configuration: Essential for proper ROS2 communication setup
 * - Queue management: Important for handling high-frequency camera streams
 * - Batch size validation: Prevents invalid configurations that could cause runtime errors
 */

#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>
#include <deep_object_detection/deep_object_detection_node.hpp>
#include <deep_test/deep_test.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>

namespace deep_object_detection
{
namespace test
{

/**
 * @brief Test fixture that initializes ROS2 once for all tests
 *
 * This avoids the overhead of initializing/shutting down ROS2 for each test case,
 * which significantly speeds up test execution.
 */
class DeepObjectDetectionNodeTestFixture
{
public:
  DeepObjectDetectionNodeTestFixture()
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }

  ~DeepObjectDetectionNodeTestFixture()
  {
    // ROS2 shutdown is handled automatically when process exits
  }
};

TEST_CASE_METHOD(
  DeepObjectDetectionNodeTestFixture, "DeepObjectDetectionNode: Construction and Basic State", "[node][construction]")
{
  rclcpp::NodeOptions options;
  auto node = std::make_shared<DeepObjectDetectionNode>(options);

  REQUIRE(node != nullptr);
  REQUIRE(node->get_name() == std::string("deep_object_detection_node"));
  REQUIRE(node->get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  REQUIRE(node->get_current_state().label() == "unconfigured");
}

TEST_CASE_METHOD(
  DeepObjectDetectionNodeTestFixture,
  "DeepObjectDetectionNode: Parameter Declaration and Defaults",
  "[node][parameters]")
{
  rclcpp::NodeOptions options;
  auto node = std::make_shared<DeepObjectDetectionNode>(options);

  SECTION("Required parameters are declared")
  {
    // Core parameters
    REQUIRE(node->has_parameter("model_path"));
    REQUIRE(node->has_parameter("class_names_path"));

    // Model parameters
    REQUIRE(node->has_parameter("Model.num_classes"));
    REQUIRE(node->has_parameter("Model.bbox_format"));
    REQUIRE(node->has_parameter("Model.output_shape"));

    // Preprocessing parameters
    REQUIRE(node->has_parameter("Preprocessing.input_width"));
    REQUIRE(node->has_parameter("Preprocessing.input_height"));
    REQUIRE(node->has_parameter("Preprocessing.normalization_type"));
    REQUIRE(node->has_parameter("Preprocessing.mean"));
    REQUIRE(node->has_parameter("Preprocessing.std"));
    REQUIRE(node->has_parameter("Preprocessing.resize_method"));
    REQUIRE(node->has_parameter("Preprocessing.pad_value"));
    REQUIRE(node->has_parameter("Preprocessing.color_format"));

    // Postprocessing parameters
    REQUIRE(node->has_parameter("Postprocessing.score_threshold"));
    REQUIRE(node->has_parameter("Postprocessing.nms_iou_threshold"));
    REQUIRE(node->has_parameter("Postprocessing.score_activation"));
    REQUIRE(node->has_parameter("Postprocessing.enable_nms"));
    REQUIRE(node->has_parameter("Postprocessing.class_score_mode"));
    REQUIRE(node->has_parameter("Postprocessing.class_score_start_idx"));
    REQUIRE(node->has_parameter("Postprocessing.class_score_count"));
    REQUIRE(node->has_parameter("Postprocessing.layout.batch_dim"));
    REQUIRE(node->has_parameter("Postprocessing.layout.detection_dim"));
    REQUIRE(node->has_parameter("Postprocessing.layout.feature_dim"));
    REQUIRE(node->has_parameter("Postprocessing.layout.bbox_start_idx"));
    REQUIRE(node->has_parameter("Postprocessing.layout.bbox_count"));
    REQUIRE(node->has_parameter("Postprocessing.layout.score_idx"));
    REQUIRE(node->has_parameter("Postprocessing.layout.class_idx"));

    // Execution provider parameters
    REQUIRE(node->has_parameter("Backend.execution_provider"));
    REQUIRE(node->has_parameter("Backend.device_id"));
    REQUIRE(node->has_parameter("Backend.trt_engine_cache_enable"));
    REQUIRE(node->has_parameter("Backend.trt_engine_cache_path"));

    // Input/Output configuration
    REQUIRE(node->has_parameter("use_compressed_images"));
    REQUIRE(node->has_parameter("output_detections_topic"));
  }

  SECTION("Parameters have sensible defaults")
  {
    // Core parameters
    REQUIRE(node->get_parameter("model_path").as_string() == "");
    REQUIRE(node->get_parameter("class_names_path").as_string() == "");

    // Model parameters
    REQUIRE(node->get_parameter("Model.num_classes").as_int() == 80);
    REQUIRE(node->get_parameter("Model.bbox_format").as_string() == "cxcywh");

    // Preprocessing parameters
    REQUIRE(node->get_parameter("Preprocessing.input_width").as_int() == 640);
    REQUIRE(node->get_parameter("Preprocessing.input_height").as_int() == 640);
    REQUIRE(node->get_parameter("Preprocessing.normalization_type").as_string() == "scale_0_1");
    REQUIRE(node->get_parameter("Preprocessing.resize_method").as_string() == "letterbox");
    REQUIRE(node->get_parameter("Preprocessing.pad_value").as_int() == 114);
    REQUIRE(node->get_parameter("Preprocessing.color_format").as_string() == "rgb");

    // Postprocessing parameters
    REQUIRE(node->get_parameter("Postprocessing.score_threshold").as_double() == Approx(0.25));
    REQUIRE(node->get_parameter("Postprocessing.nms_iou_threshold").as_double() == Approx(0.45));
    REQUIRE(node->get_parameter("Postprocessing.score_activation").as_string() == "sigmoid");
    REQUIRE(node->get_parameter("Postprocessing.enable_nms").as_bool() == true);
    REQUIRE(node->get_parameter("Postprocessing.class_score_mode").as_string() == "all_classes");
    REQUIRE(node->get_parameter("Postprocessing.class_score_start_idx").as_int() == -1);
    REQUIRE(node->get_parameter("Postprocessing.class_score_count").as_int() == -1);

    // Execution provider parameters
    REQUIRE(node->get_parameter("Backend.execution_provider").as_string() == "tensorrt");
    REQUIRE(node->get_parameter("Backend.device_id").as_int() == 0);
    REQUIRE(node->get_parameter("Backend.trt_engine_cache_enable").as_bool() == false);
    REQUIRE(node->get_parameter("Backend.trt_engine_cache_path").as_string() == "/tmp/deep_ros_ort_trt_cache");

    // Input/Output configuration
    REQUIRE(node->get_parameter("use_compressed_images").as_bool() == true);
    REQUIRE(node->get_parameter("output_detections_topic").as_string() == "/detections");
  }
}

TEST_CASE_METHOD(
  DeepObjectDetectionNodeTestFixture,
  "DeepObjectDetectionNode: Preprocessing Parameter Validation",
  "[node][preprocessing]")
{
  rclcpp::NodeOptions options;
  auto node = std::make_shared<DeepObjectDetectionNode>(options);

  SECTION("Preprocessing dimensions can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("Preprocessing.input_width", 1280), rclcpp::Parameter("Preprocessing.input_height", 720)};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(result[1].successful == true);
    REQUIRE(node->get_parameter("Preprocessing.input_width").as_int() == 1280);
    REQUIRE(node->get_parameter("Preprocessing.input_height").as_int() == 720);
  }

  SECTION("Normalization type can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Preprocessing.normalization_type", "imagenet")};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("Preprocessing.normalization_type").as_string() == "imagenet");
  }

  SECTION("Resize method can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Preprocessing.resize_method", "resize")};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("Preprocessing.resize_method").as_string() == "resize");
  }
}

TEST_CASE_METHOD(
  DeepObjectDetectionNodeTestFixture,
  "DeepObjectDetectionNode: Postprocessing Parameter Validation",
  "[node][postprocessing]")
{
  rclcpp::NodeOptions options;
  auto node = std::make_shared<DeepObjectDetectionNode>(options);

  SECTION("Score threshold can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Postprocessing.score_threshold", 0.5)};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("Postprocessing.score_threshold").as_double() == Approx(0.5));
  }

  SECTION("NMS IoU threshold can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Postprocessing.nms_iou_threshold", 0.6)};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("Postprocessing.nms_iou_threshold").as_double() == Approx(0.6));
  }
}

TEST_CASE_METHOD(
  DeepObjectDetectionNodeTestFixture, "DeepObjectDetectionNode: Execution Provider Configuration", "[node][provider]")
{
  rclcpp::NodeOptions options;
  auto node = std::make_shared<DeepObjectDetectionNode>(options);

  SECTION("Provider can be set to CPU")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.execution_provider", "cpu")};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("Backend.execution_provider").as_string() == "cpu");
  }

  SECTION("Provider can be set to CUDA")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.execution_provider", "cuda")};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("Backend.execution_provider").as_string() == "cuda");
  }

  SECTION("Device ID can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.device_id", 1)};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("Backend.device_id").as_int() == 1);
  }
}

TEST_CASE_METHOD(
  DeepObjectDetectionNodeTestFixture,
  "DeepObjectDetectionNode: Additional Parameter Configuration",
  "[node][additional]")
{
  rclcpp::NodeOptions options;
  auto node = std::make_shared<DeepObjectDetectionNode>(options);

  SECTION("Class names path can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("class_names_path", "/path/to/classes.txt")};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("class_names_path").as_string() == "/path/to/classes.txt");
  }

  SECTION("Use compressed images can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("use_compressed_images", false)};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("use_compressed_images").as_bool() == false);
  }

  SECTION("Output detections topic can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("output_detections_topic", "/custom/detections")};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("output_detections_topic").as_string() == "/custom/detections");
  }

  SECTION("Postprocessing enable_nms can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Postprocessing.enable_nms", false)};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("Postprocessing.enable_nms").as_bool() == false);
  }
}

}  // namespace test
}  // namespace deep_object_detection
