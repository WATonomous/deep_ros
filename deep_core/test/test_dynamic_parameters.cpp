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

#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>
#include <deep_core/deep_node_base.hpp>
#include <deep_test/deep_test.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace deep_ros
{
namespace test
{

class TestDynamicReconfigurationNode : public DeepNodeBase
{
public:
  explicit TestDynamicReconfigurationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : DeepNodeBase("test_dynamic_node", options)
  {
    // Declare test parameters
    declare_parameter("some_other_param", "default_value");
  }

  using DeepNodeBase::get_current_state;
  using DeepNodeBase::is_model_loaded;
  using DeepNodeBase::is_plugin_loaded;

protected:
  CallbackReturn on_configure_impl(const rclcpp_lifecycle::State & /*state*/) override
  {
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate_impl(const rclcpp_lifecycle::State & /*state*/) override
  {
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate_impl(const rclcpp_lifecycle::State & /*state*/) override
  {
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_cleanup_impl(const rclcpp_lifecycle::State & /*state*/) override
  {
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_shutdown_impl(const rclcpp_lifecycle::State & /*state*/) override
  {
    return CallbackReturn::SUCCESS;
  }
};

TEST_CASE_METHOD(deep_ros::test::TestExecutorFixture, "Dynamic model reconfiguration", "[dynamic_reconfig]")
{
  // Create node with test plugin enabled by default
  rclcpp::NodeOptions options;
  options.parameter_overrides({{"Backend.plugin", "test_backend"}, {"model_path", ""}, {"Bond.enable", false}});

  auto test_node = std::make_shared<TestDynamicReconfigurationNode>(options);
  add_node(test_node);
  start_spinning();

  SECTION("Parameter callback is registered")
  {
    REQUIRE(test_node != nullptr);
    REQUIRE(test_node->get_name() == std::string("test_dynamic_node"));

    // Node should be in unconfigured state initially
    REQUIRE(test_node->get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  }

  SECTION("Model parameter change allowed when not active (no callback registered)")
  {
    // Try to set model_path while in unconfigured state
    auto parameters = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/some/model/path.onnx")};

    auto result = test_node->set_parameters(parameters);

    // Should succeed because callback is not registered until activation
    REQUIRE(result[0].successful == true);

    // Verify the parameter was set
    REQUIRE(test_node->get_parameter("model_path").as_string() == "/some/model/path.onnx");
  }

  SECTION("Configuration fails with invalid plugin")
  {
    // First disable the test plugin and set an invalid one
    auto invalid_plugin_params =
      std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.plugin", "nonexistent_plugin")};
    auto invalid_result = test_node->set_parameters(invalid_plugin_params);
    REQUIRE(invalid_result[0].successful == true);  // Parameter setting succeeds

    // Configure the node first (this will try to load the invalid plugin and should fail)
    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);  // Configuration failed

    // Verify no plugin is loaded
    REQUIRE(test_node->is_plugin_loaded() == false);

    // Node should remain in unconfigured state - cannot proceed to activation
    REQUIRE(test_node->get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  }

  SECTION("Model parameter change succeeds with plugin")
  {
    // Configure the node (test plugin already set in options)
    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

    // Verify plugin loaded during configure
    REQUIRE(test_node->is_plugin_loaded() == true);

    // Activate the node
    auto activate_result = test_node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

    // Now try to change model_path while active
    auto parameters = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/some/model/path.onnx")};

    auto result = test_node->set_parameters(parameters);

    // Should succeed with the test plugin loaded
    REQUIRE(result[0].successful == true);
    REQUIRE(test_node->is_model_loaded() == true);
  }

  SECTION("Empty model path always rejected")
  {
    // Configure and activate the node (plugin will be loaded)
    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
    REQUIRE(test_node->is_plugin_loaded() == true);

    auto activate_result = test_node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

    // Set empty model path - should always be rejected
    auto parameters = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "")};

    auto result = test_node->set_parameters(parameters);

    // Should fail because empty paths are not allowed
    REQUIRE(result[0].successful == false);
    REQUIRE(result[0].reason.find("Cannot set empty model path") != std::string::npos);
    REQUIRE(test_node->is_model_loaded() == false);
  }

  SECTION("Model change with same path still processes")
  {
    // Ensure plugin is set (test_node should already have test_backend from options)
    // Configure and activate the node with plugin
    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
    REQUIRE(test_node->is_plugin_loaded() == true);

    auto activate_result = test_node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

    // Set model path twice with same value
    auto parameters = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/same/model/path.onnx")};

    // First attempt
    auto result1 = test_node->set_parameters(parameters);

    // Second attempt with same path should still process
    auto result2 = test_node->set_parameters(parameters);

    // Both should succeed with the plugin loaded
    REQUIRE(result1[0].successful == true);
    REQUIRE(result2[0].successful == true);
    REQUIRE(test_node->is_model_loaded() == true);
  }

  SECTION("Non-model parameters are ignored")
  {
    // Configure and activate the node (plugin will be loaded)
    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
    REQUIRE(test_node->is_plugin_loaded() == true);

    auto activate_result = test_node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

    // Capture initial state
    auto initial_model_loaded = test_node->is_model_loaded();
    auto initial_plugin_loaded = test_node->is_plugin_loaded();
    auto initial_lifecycle_state = test_node->get_current_state().id();

    // Set a non-model parameter
    auto parameters = std::vector<rclcpp::Parameter>{rclcpp::Parameter("some_other_param", "some_value")};

    auto result = test_node->set_parameters(parameters);

    // Should succeed since it's not handled by the callback
    REQUIRE(result[0].successful == true);

    // Verify nothing important changed
    REQUIRE(test_node->is_model_loaded() == initial_model_loaded);
    REQUIRE(test_node->is_plugin_loaded() == initial_plugin_loaded);
    REQUIRE(test_node->get_current_state().id() == initial_lifecycle_state);

    // Verify the parameter was actually set
    REQUIRE(test_node->get_parameter("some_other_param").as_string() == "some_value");
  }

  SECTION("Multiple parameters with model_path")
  {
    // Configure and activate the node with plugin
    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
    REQUIRE(test_node->is_plugin_loaded() == true);

    auto activate_result = test_node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

    // Set multiple parameters including model_path
    auto parameters = std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("model_path", "/new/model/path.onnx"), rclcpp::Parameter("some_other_param", 42)};

    auto result = test_node->set_parameters(parameters);

    // Should handle both parameters
    REQUIRE(result.size() == 2);

    // Both parameters should succeed - model_path succeeds with plugin loaded
    bool model_path_succeeded = false;
    bool other_param_succeeded = false;
    for (const auto & res : result) {
      if (res.successful) {
        if (res.reason.find("model_path") != std::string::npos || res.reason.empty()) {
          model_path_succeeded = true;
        } else {
          other_param_succeeded = true;
        }
      }
    }
    REQUIRE(test_node->is_model_loaded() == true);
  }
}

}  // namespace test
}  // namespace deep_ros
