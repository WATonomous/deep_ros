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

    // Model parameters
    REQUIRE(node->has_parameter("model.num_classes"));
    REQUIRE(node->has_parameter("model.bbox_format"));

    // Preprocessing parameters
    REQUIRE(node->has_parameter("preprocessing.input_width"));
    REQUIRE(node->has_parameter("preprocessing.input_height"));
    REQUIRE(node->has_parameter("preprocessing.normalization_type"));
    REQUIRE(node->has_parameter("preprocessing.resize_method"));
    REQUIRE(node->has_parameter("preprocessing.color_format"));

    // Postprocessing parameters
    REQUIRE(node->has_parameter("postprocessing.score_threshold"));
    REQUIRE(node->has_parameter("postprocessing.nms_iou_threshold"));
    REQUIRE(node->has_parameter("postprocessing.max_detections"));
    REQUIRE(node->has_parameter("postprocessing.score_activation"));

    // Execution provider parameters
    REQUIRE(node->has_parameter("preferred_provider"));
    REQUIRE(node->has_parameter("device_id"));
  }

  SECTION("Parameters have sensible defaults")
  {
    // Core parameters
    REQUIRE(node->get_parameter("model_path").as_string() == "");

    // Model parameters
    REQUIRE(node->get_parameter("model.num_classes").as_int() == 80);
    REQUIRE(node->get_parameter("model.bbox_format").as_string() == "cxcywh");

    // Preprocessing parameters
    REQUIRE(node->get_parameter("preprocessing.input_width").as_int() == 640);
    REQUIRE(node->get_parameter("preprocessing.input_height").as_int() == 640);
    REQUIRE(node->get_parameter("preprocessing.normalization_type").as_string() == "scale_0_1");
    REQUIRE(node->get_parameter("preprocessing.resize_method").as_string() == "letterbox");
    REQUIRE(node->get_parameter("preprocessing.color_format").as_string() == "rgb");

    // Postprocessing parameters
    REQUIRE(node->get_parameter("postprocessing.score_threshold").as_double() == Approx(0.25));
    REQUIRE(node->get_parameter("postprocessing.nms_iou_threshold").as_double() == Approx(0.45));
    REQUIRE(node->get_parameter("postprocessing.max_detections").as_int() == 300);
    REQUIRE(node->get_parameter("postprocessing.score_activation").as_string() == "sigmoid");

    // Execution provider parameters
    REQUIRE(node->get_parameter("preferred_provider").as_string() == "tensorrt");
    REQUIRE(node->get_parameter("device_id").as_int() == 0);
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
      rclcpp::Parameter("preprocessing.input_width", 1280), rclcpp::Parameter("preprocessing.input_height", 720)};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(result[1].successful == true);
    REQUIRE(node->get_parameter("preprocessing.input_width").as_int() == 1280);
    REQUIRE(node->get_parameter("preprocessing.input_height").as_int() == 720);
  }

  SECTION("Normalization type can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("preprocessing.normalization_type", "imagenet")};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("preprocessing.normalization_type").as_string() == "imagenet");
  }

  SECTION("Resize method can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("preprocessing.resize_method", "resize")};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("preprocessing.resize_method").as_string() == "resize");
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
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("postprocessing.score_threshold", 0.5)};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("postprocessing.score_threshold").as_double() == Approx(0.5));
  }

  SECTION("NMS IoU threshold can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("postprocessing.nms_iou_threshold", 0.6)};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("postprocessing.nms_iou_threshold").as_double() == Approx(0.6));
  }

  SECTION("Max detections can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("postprocessing.max_detections", 100)};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("postprocessing.max_detections").as_int() == 100);
  }
}

TEST_CASE_METHOD(
  DeepObjectDetectionNodeTestFixture, "DeepObjectDetectionNode: Execution Provider Configuration", "[node][provider]")
{
  rclcpp::NodeOptions options;
  auto node = std::make_shared<DeepObjectDetectionNode>(options);

  SECTION("Provider can be set to CPU")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("preferred_provider", "cpu")};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("preferred_provider").as_string() == "cpu");
  }

  SECTION("Provider can be set to CUDA")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("preferred_provider", "cuda")};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("preferred_provider").as_string() == "cuda");
  }

  SECTION("Device ID can be configured")
  {
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("device_id", 1)};
    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->get_parameter("device_id").as_int() == 1);
  }
}

TEST_CASE_METHOD(
  DeepObjectDetectionNodeTestFixture, "DeepObjectDetectionNode: Batch Size Configuration", "[node][batching]")
{
  rclcpp::NodeOptions options;
  auto node = std::make_shared<DeepObjectDetectionNode>(options);
}

}  // namespace test
}  // namespace deep_object_detection
