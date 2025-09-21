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
  // Create node with initial parameters
  rclcpp::NodeOptions options;
  options.parameter_overrides({{"Backend.plugin", ""}, {"model_path", ""}, {"Bond.enable", false}});

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

  SECTION("Model parameter change rejected when not active")
  {
    // Try to set model_path while in unconfigured state
    auto parameters = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/some/model/path.onnx")};

    auto result = test_node->set_parameters(parameters);

    // Should fail because node is not active
    REQUIRE(result[0].successful == false);
    REQUIRE(result[0].reason.find("Node must be active") != std::string::npos);
  }

  SECTION("Model parameter change allowed when active")
  {
    // Configure the node first
    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

    // Activate the node
    auto activate_result = test_node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

    // Now try to change model_path while active
    auto parameters = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/some/model/path.onnx")};

    auto result = test_node->set_parameters(parameters);

    // Should fail gracefully because no plugin is loaded, but parameter change should be allowed
    REQUIRE(result[0].successful == false);
    REQUIRE(result[0].reason.find("Failed to load new model") != std::string::npos);
  }

  SECTION("Empty model path always rejected")
  {
    // Configure and activate the node
    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

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

  SECTION("Model change with same path is ignored")
  {
    // Configure and activate the node
    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

    auto activate_result = test_node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

    // Set model path twice with same value
    auto parameters = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/same/model/path.onnx")};

    // First attempt
    auto result1 = test_node->set_parameters(parameters);

    // Second attempt with same path should still process (though may fail due to no plugin)
    auto result2 = test_node->set_parameters(parameters);

    // Both should handle the parameter change attempt
    REQUIRE(result1.size() == 1);
    REQUIRE(result2.size() == 1);
  }

  SECTION("Non-model parameters are ignored")
  {
    // Configure and activate the node
    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

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
    // Configure and activate the node
    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

    auto activate_result = test_node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

    // Set multiple parameters including model_path
    auto parameters = std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("model_path", "/new/model/path.onnx"), rclcpp::Parameter("some_other_param", 42)};

    auto result = test_node->set_parameters(parameters);

    // Should handle both parameters
    REQUIRE(result.size() == 2);

    // The parameter callback handles model_path and fails it
    // The other parameter may also fail depending on ROS parameter behavior
    bool model_path_failed = false;
    for (const auto & res : result) {
      if (!res.successful && res.reason.find("Failed to load new model") != std::string::npos) {
        model_path_failed = true;
        break;
      }
    }
    REQUIRE(model_path_failed == true);
  }
}

}  // namespace test
}  // namespace deep_ros
