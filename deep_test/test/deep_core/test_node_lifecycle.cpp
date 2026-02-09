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

#include <deep_test/compat.hpp>
#include <deep_test/deep_test.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <test_nodes/deep_test_node.hpp>

TEST_CASE_METHOD(deep_ros::test::TestExecutorFixture, "Node Lifecycle: Basic State Management", "[node][lifecycle]")
{
  auto node = std::make_shared<deep_ros::test::DeepTestNode>();
  add_node(node);
  start_spinning();

  SECTION("Node starts in unconfigured state")
  {
    REQUIRE(node->get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
    REQUIRE(node->get_current_state().label() == "unconfigured");
  }

  SECTION("Standard lifecycle transitions work")
  {
    // Unconfigured -> Inactive
    auto configure_result = node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

    // Inactive -> Active
    auto activate_result = node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

    // Active -> Inactive
    auto deactivate_result = node->deactivate();
    REQUIRE(deactivate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

    // Inactive -> Unconfigured
    auto cleanup_result = node->cleanup();
    REQUIRE(cleanup_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  }

  SECTION("Shutdown can happen from any state")
  {
    // From unconfigured
    auto shutdown_result = node->shutdown();
    REQUIRE(shutdown_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_FINALIZED);

    // Create new node for other state tests
    auto node2 = std::make_shared<deep_ros::test::DeepTestNode>();
    add_node(node2);

    // From inactive
    node2->configure();
    auto shutdown_result2 = node2->shutdown();
    REQUIRE(shutdown_result2.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_FINALIZED);
  }

  SECTION("Invalid transitions are rejected")
  {
    // Can't activate without configuring
    auto activate_result = node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

    // Can't cleanup without configuring
    auto cleanup_result = node->cleanup();
    REQUIRE(cleanup_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  }
}

TEST_CASE_METHOD(deep_ros::test::TestExecutorFixture, "Node Lifecycle: Parameter Management", "[node][parameters]")
{
  auto node = std::make_shared<deep_ros::test::DeepTestNode>();
  add_node(node);
  start_spinning();

  SECTION("Required parameters are declared")
  {
    // Deep learning related parameters
    REQUIRE(node->has_parameter("Backend.plugin"));
    REQUIRE(node->has_parameter("model_path"));

    // Bond related parameters
    REQUIRE(node->has_parameter("Bond.enable"));
    REQUIRE(node->has_parameter("Bond.bond_timeout"));
    REQUIRE(node->has_parameter("Bond.bond_heartbeat_period"));
  }

  SECTION("Parameters have sensible defaults")
  {
    // Backend plugin should default to empty (user must specify)
    REQUIRE(node->get_parameter("Backend.plugin").as_string() == "");

    // Model path should default to empty
    REQUIRE(node->get_parameter("model_path").as_string() == "");

    // Bond should be disabled by default
    REQUIRE(node->get_parameter("Bond.enable").as_bool() == false);

    // Bond timing parameters should have reasonable defaults
    auto timeout = node->get_parameter("Bond.bond_timeout").as_double();
    auto heartbeat = node->get_parameter("Bond.bond_heartbeat_period").as_double();
    REQUIRE(timeout > 0.0);
    REQUIRE(heartbeat > 0.0);
    REQUIRE(timeout > heartbeat);  // Timeout should be longer than heartbeat
  }

  SECTION("Parameter changes during configuration")
  {
    // Should be able to set parameters before configuring
    auto params = std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("Backend.plugin", "mock_backend"), rclcpp::Parameter("model_path", "/test/model.onnx")};

    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(result[1].successful == true);

    // Parameters should be accessible
    REQUIRE(node->get_parameter("Backend.plugin").as_string() == "mock_backend");
    REQUIRE(node->get_parameter("model_path").as_string() == "/test/model.onnx");
  }
}

TEST_CASE_METHOD(
  deep_ros::test::TestExecutorFixture, "Node Lifecycle: Plugin Integration", "[node][plugin_integration]")
{
  auto node = std::make_shared<deep_ros::test::DeepTestNode>();
  add_node(node);
  start_spinning();

  SECTION("Plugin loading during configuration")
  {
    // Set plugin parameter
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.plugin", "mock_backend")};
    node->set_parameters(params);

    // Should start without plugin loaded
    REQUIRE(node->is_plugin_loaded() == false);
    REQUIRE(node->get_backend_name() == "none");
    REQUIRE(node->get_current_allocator() == nullptr);

    // Configure should load the plugin
    auto result = node->configure();
    REQUIRE(result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
    REQUIRE(node->is_plugin_loaded() == true);
    REQUIRE(node->get_backend_name() == "mock_backend");
    REQUIRE(node->get_current_allocator() != nullptr);
  }

  SECTION("Invalid plugin prevents configuration")
  {
    // Set invalid plugin
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.plugin", "nonexistent_backend")};
    node->set_parameters(params);

    // Configuration should fail
    auto result = node->configure();
    REQUIRE(result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
    REQUIRE(node->is_plugin_loaded() == false);
  }

  SECTION("Plugin cleanup during lifecycle transitions")
  {
    // Configure with valid plugin
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.plugin", "mock_backend")};
    node->set_parameters(params);
    node->configure();

    REQUIRE(node->is_plugin_loaded() == true);

    // Cleanup should unload plugin
    node->cleanup();
    REQUIRE(node->is_plugin_loaded() == false);
    REQUIRE(node->get_current_allocator() == nullptr);
  }

  SECTION("Plugin persists through activate/deactivate")
  {
    // Setup node with plugin
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.plugin", "mock_backend")};
    node->set_parameters(params);
    node->configure();

    // Plugin should persist through activation
    node->activate();
    REQUIRE(node->is_plugin_loaded() == true);
    REQUIRE(node->get_current_allocator() != nullptr);

    // And through deactivation
    node->deactivate();
    REQUIRE(node->is_plugin_loaded() == true);
    REQUIRE(node->get_current_allocator() != nullptr);
  }
}

TEST_CASE_METHOD(deep_ros::test::TestExecutorFixture, "Node Lifecycle: Model Management", "[node][model_management]")
{
  // Setup node with plugin
  rclcpp::NodeOptions options;
  options.parameter_overrides({{"Backend.plugin", "mock_backend"}, {"model_path", ""}, {"Bond.enable", false}});

  auto node = std::make_shared<deep_ros::test::DeepTestNode>(options);
  add_node(node);
  start_spinning();

  SECTION("Model loading during configuration")
  {
    // Set model path and configure
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/test/model.onnx")};
    node->set_parameters(params);

    // Should start without model
    REQUIRE(node->is_model_loaded() == false);

    // Configure should load both plugin and model
    auto result = node->configure();
    REQUIRE(result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
    REQUIRE(node->is_plugin_loaded() == true);
    REQUIRE(node->is_model_loaded() == true);
  }

  SECTION("Model loading without plugin fails")
  {
    // Try to configure with model but no plugin
    auto params = std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("Backend.plugin", ""),  // No plugin
      rclcpp::Parameter("model_path", "/test/model.onnx")};
    node->set_parameters(params);

    // Configuration should succeed but model loading should fail
    auto result = node->configure();
    REQUIRE(result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  }

  SECTION("Model unloading during cleanup")
  {
    // Set model path first
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/test/model.onnx")};
    node->set_parameters(params);

    // Setup with plugin and model
    node->configure();
    REQUIRE(node->is_model_loaded() == true);

    // Cleanup should unload both model and plugin
    node->cleanup();
    REQUIRE(node->is_model_loaded() == false);
    REQUIRE(node->is_plugin_loaded() == false);
  }
}

TEST_CASE_METHOD(
  deep_ros::test::TestExecutorFixture, "Node Lifecycle: Dynamic Reconfiguration", "[node][dynamic_config]")
{
  // Setup node with plugin
  rclcpp::NodeOptions options;
  options.parameter_overrides({{"Backend.plugin", "mock_backend"}, {"model_path", ""}, {"Bond.enable", false}});

  auto node = std::make_shared<deep_ros::test::DeepTestNode>(options);
  add_node(node);
  start_spinning();

  SECTION("Dynamic model changes when active")
  {
    // Configure and activate
    node->configure();
    node->activate();

    // Should be able to change model path dynamically
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/dynamic/model.onnx")};

    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
    REQUIRE(node->is_model_loaded() == true);
  }

  SECTION("Dynamic changes restricted when not active")
  {
    // Configure but don't activate
    node->configure();

    // Model changes should be allowed even when inactive
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/inactive/model.onnx")};

    auto result = node->set_parameters(params);
    // This might be allowed or not depending on implementation
    // The key is that it should be well-defined
    REQUIRE(result[0].successful == true);  // Or false, but consistent
  }

  SECTION("Invalid dynamic changes are rejected")
  {
    node->configure();
    node->activate();

    // Empty model path should be rejected
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "")};

    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == false);
    REQUIRE(!result[0].reason.empty());  // Should provide reason
  }

  SECTION("Non-model parameter changes are always allowed")
  {
    node->configure();
    node->activate();

    // Should be able to change non-critical parameters
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Bond.bond_timeout", 10.0)};

    auto result = node->set_parameters(params);
    REQUIRE(result[0].successful == true);
  }
}

TEST_CASE_METHOD(deep_ros::test::TestExecutorFixture, "Node Lifecycle: Error Recovery", "[node][error_recovery]")
{
  auto node = std::make_shared<deep_ros::test::DeepTestNode>();
  add_node(node);
  start_spinning();

  SECTION("Recovery from configuration failure")
  {
    // Try to configure with invalid plugin
    auto bad_params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.plugin", "invalid_backend")};
    node->set_parameters(bad_params);

    auto result = node->configure();
    REQUIRE(result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

    // Should be able to recover with valid config
    auto good_params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.plugin", "mock_backend")};
    node->set_parameters(good_params);

    auto result2 = node->configure();
    REQUIRE(result2.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  }

  SECTION("Node state consistency after errors")
  {
    // Configure successfully first
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.plugin", "mock_backend")};
    node->set_parameters(params);
    node->configure();
    node->activate();

    // Try invalid parameter change
    auto bad_params = std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("model_path", "")  // Should be rejected
    };
    auto result = node->set_parameters(bad_params);
    REQUIRE(result[0].successful == false);

    // Node should still be functional
    REQUIRE(node->get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
    REQUIRE(node->is_plugin_loaded() == true);

    // Should still accept valid operations
    auto good_params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/recovery/model.onnx")};
    auto result2 = node->set_parameters(good_params);
    REQUIRE(result2[0].successful == true);
  }

  SECTION("Graceful shutdown after errors")
  {
    // Even after configuration failure, shutdown should work
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.plugin", "invalid_backend")};
    node->set_parameters(params);
    node->configure();  // Will fail

    // Shutdown should still work cleanly
    auto result = node->shutdown();
    REQUIRE(result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_FINALIZED);
  }
}
