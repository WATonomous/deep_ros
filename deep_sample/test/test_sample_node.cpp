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

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch.hpp>
#include <deep_sample/sample_inference_node.hpp>
#include <deep_test/deep_test.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

namespace deep_sample
{
namespace test
{

class SampleNodeTestFixture
{
public:
  SampleNodeTestFixture()
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }

  ~SampleNodeTestFixture()
  {
    // Clean up
  }
};

TEST_CASE_METHOD(SampleNodeTestFixture, "SampleInferenceNode construction", "[sample][node]")
{
  rclcpp::NodeOptions options;
  auto node = std::make_shared<SampleInferenceNode>(options);

  REQUIRE(node != nullptr);
  REQUIRE(node->get_name() == std::string("sample_inference_node"));
}

TEST_CASE_METHOD(SampleNodeTestFixture, "SampleInferenceNode lifecycle transitions", "[sample][lifecycle]")
{
  rclcpp::NodeOptions options;
  auto node = std::make_shared<SampleInferenceNode>(options);

  SECTION("Configure without backend fails")
  {
    auto result = node->configure();
    // Should succeed configure even without plugin/model for this simple test
    REQUIRE(result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  }

  SECTION("Activate without backend/model fails")
  {
    node->configure();
    auto result = node->activate();
    // Should fail activation without plugin and model loaded
    REQUIRE(result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  }

  SECTION("Full lifecycle with parameters")
  {
    // Create a fresh node with parameters passed via NodeOptions
    rclcpp::NodeOptions options;
    options.append_parameter_override("Backend.plugin", "onnxruntime_cpu");
    options.append_parameter_override("model_path", TEST_MODEL_PATH);

    auto test_node = std::make_shared<SampleInferenceNode>(options);

    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

    auto activate_result = test_node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

    auto deactivate_result = test_node->deactivate();
    REQUIRE(deactivate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

    auto cleanup_result = test_node->cleanup();
    REQUIRE(cleanup_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  }
}

TEST_CASE_METHOD(SampleNodeTestFixture, "SampleInferenceNode parameter handling", "[sample][params]")
{
  rclcpp::NodeOptions options;
  auto node = std::make_shared<SampleInferenceNode>(options);

  node->configure();

  SECTION("Default topic names")
  {
    auto input_topic = node->get_parameter("input_topic").as_string();
    auto output_topic = node->get_parameter("output_topic").as_string();

    REQUIRE(input_topic == "/camera/image_raw");
    REQUIRE(output_topic == "/inference/output");
  }

  SECTION("Custom topic names")
  {
    node->set_parameter(rclcpp::Parameter("input_topic", "/custom/input"));
    node->set_parameter(rclcpp::Parameter("output_topic", "/custom/output"));

    auto input_topic = node->get_parameter("input_topic").as_string();
    auto output_topic = node->get_parameter("output_topic").as_string();

    REQUIRE(input_topic == "/custom/input");
    REQUIRE(output_topic == "/custom/output");
  }
}

}  // namespace test
}  // namespace deep_sample
