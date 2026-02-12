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
#include <utility>
#include <vector>

#include <deep_core/types/tensor.hpp>
#include <deep_test/compat.hpp>
#include <deep_test/deep_test.hpp>
#include <test_nodes/deep_test_node.hpp>

TEST_CASE_METHOD(
  deep_ros::test::TestExecutorFixture, "Integration: Complete Inference Pipeline", "[integration][inference]")
{
  // Setup node with everything configured
  rclcpp::NodeOptions options;
  options.parameter_overrides(
    {{"Backend.plugin", "mock_backend"}, {"model_path", "/test/model.onnx"}, {"Bond.enable", false}});

  auto node = std::make_shared<deep_ros::test::DeepTestNode>(options);
  add_node(node);
  start_spinning();

  SECTION("Full pipeline: configure -> activate -> inference")
  {
    // Step 1: Configure (loads plugin and model)
    auto configure_result = node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
    REQUIRE(node->is_plugin_loaded() == true);
    REQUIRE(node->is_model_loaded() == true);

    // Step 2: Activate
    auto activate_result = node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

    // Step 3: Run inference
    auto allocator = node->get_current_allocator();
    REQUIRE(allocator != nullptr);

    // Create realistic input tensor
    std::vector<size_t> shape = {1, 3, 224, 224};  // Typical image input
    deep_ros::Tensor input(shape, deep_ros::DataType::FLOAT32, allocator);

    // Fill with test data (normalized image values)
    auto data = input.data_as<float>();
    for (size_t i = 0; i < input.size(); ++i) {
      data[i] = static_cast<float>(i % 256) / 255.0f;  // 0-1 range
    }

    // Run inference
    auto output = node->run_inference(input);

    // Verify output
    REQUIRE(output.data() != nullptr);
    REQUIRE(output.size() > 0);
    REQUIRE(output.dtype() == deep_ros::DataType::FLOAT32);
    REQUIRE(output.shape() == shape);  // Mock returns same shape
  }

  SECTION("Multiple inference calls work correctly")
  {
    // Setup pipeline
    node->configure();
    node->activate();

    auto allocator = node->get_current_allocator();

    // Run multiple inferences
    for (int i = 0; i < 5; ++i) {
      std::vector<size_t> shape = {1, 3, 32, 32};
      deep_ros::Tensor input(shape, deep_ros::DataType::FLOAT32, allocator);

      auto data = input.data_as<float>();
      for (size_t j = 0; j < input.size(); ++j) {
        data[j] = static_cast<float>(i * 100 + j);
      }

      auto output = node->run_inference(input);
      REQUIRE(output.size() == shape[0] * shape[1] * shape[2] * shape[3]);
    }
  }

  SECTION("Different tensor types work correctly")
  {
    node->configure();
    node->activate();

    auto allocator = node->get_current_allocator();

    // Test different data types
    std::vector<deep_ros::DataType> types = {
      deep_ros::DataType::FLOAT32, deep_ros::DataType::FLOAT64, deep_ros::DataType::INT32, deep_ros::DataType::UINT8};

    for (auto dtype : types) {
      std::vector<size_t> shape = {2, 3};
      deep_ros::Tensor input(shape, dtype, allocator);

      auto output = node->run_inference(input);
      REQUIRE(output.dtype() == dtype);
      REQUIRE(output.shape() == shape);
    }
  }
}

TEST_CASE_METHOD(
  deep_ros::test::TestExecutorFixture, "Integration: Dynamic Reconfiguration Workflows", "[integration][dynamic]")
{
  rclcpp::NodeOptions options;
  options.parameter_overrides(
    {{"Backend.plugin", "mock_backend"},
     {"model_path", ""},  // Start without model
     {"Bond.enable", false}});

  auto node = std::make_shared<deep_ros::test::DeepTestNode>(options);
  add_node(node);
  start_spinning();

  SECTION("Runtime model switching")
  {
    // Start with no model
    node->configure();
    node->activate();

    REQUIRE(node->is_plugin_loaded() == true);
    REQUIRE(node->is_model_loaded() == false);

    // Load first model
    auto params1 = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/models/first.onnx")};
    auto result1 = node->set_parameters(params1);
    REQUIRE(result1[0].successful == true);
    REQUIRE(node->is_model_loaded() == true);

    // Test inference with first model
    auto allocator = node->get_current_allocator();
    deep_ros::Tensor input1({2, 2}, deep_ros::DataType::FLOAT32, allocator);
    auto output1 = node->run_inference(input1);
    REQUIRE(output1.size() == 4);

    // Switch to second model
    auto params2 = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/models/second.onnx")};
    auto result2 = node->set_parameters(params2);
    REQUIRE(result2[0].successful == true);
    REQUIRE(node->is_model_loaded() == true);

    // Test inference with second model
    deep_ros::Tensor input2({3, 3}, deep_ros::DataType::FLOAT32, allocator);
    auto output2 = node->run_inference(input2);
    REQUIRE(output2.size() == 9);
  }

  SECTION("Configuration changes during operation")
  {
    node->configure();
    node->activate();

    // Load initial model
    auto params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/initial/model.onnx")};
    node->set_parameters(params);

    auto allocator = node->get_current_allocator();

    // Interleave inference and configuration changes
    for (int i = 0; i < 3; ++i) {
      // Run inference
      deep_ros::Tensor input({1, 3}, deep_ros::DataType::FLOAT32, allocator);
      auto output = node->run_inference(input);
      REQUIRE(output.size() == 3);

      // Change configuration
      auto new_params =
        std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/models/model_" + std::to_string(i) + ".onnx")};
      auto result = node->set_parameters(new_params);
      REQUIRE(result[0].successful == true);
    }
  }
}

TEST_CASE_METHOD(
  deep_ros::test::TestExecutorFixture, "Integration: Error Scenarios and Recovery", "[integration][errors]")
{
  auto node = std::make_shared<deep_ros::test::DeepTestNode>();
  add_node(node);
  start_spinning();

  SECTION("Recovery from plugin loading failure")
  {
    // Start with invalid plugin
    auto bad_params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("Backend.plugin", "nonexistent_backend")};
    node->set_parameters(bad_params);

    // Configuration should fail
    auto result = node->configure();
    REQUIRE(result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

    // Fix configuration and retry
    auto good_params = std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("Backend.plugin", "mock_backend"), rclcpp::Parameter("model_path", "/recovery/model.onnx")};
    node->set_parameters(good_params);

    // Should work now
    auto result2 = node->configure();
    REQUIRE(result2.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

    node->activate();

    // Inference should work
    auto allocator = node->get_current_allocator();
    deep_ros::Tensor input({2, 2}, deep_ros::DataType::FLOAT32, allocator);
    REQUIRE_NOTHROW(node->run_inference(input));
  }

  SECTION("Graceful handling of inference errors")
  {
    // Setup working pipeline
    auto params = std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("Backend.plugin", "mock_backend"), rclcpp::Parameter("model_path", "/test/model.onnx")};
    node->set_parameters(params);
    node->configure();
    node->activate();

    auto allocator = node->get_current_allocator();

    // Valid inference should work
    deep_ros::Tensor valid_input({2, 2}, deep_ros::DataType::FLOAT32, allocator);
    REQUIRE_NOTHROW(node->run_inference(valid_input));

    // Invalid inference should fail gracefully
    deep_ros::Tensor invalid_input;  // Empty tensor
    REQUIRE_THROWS(node->run_inference(invalid_input));

    // Node should still work after error
    deep_ros::Tensor recovery_input({3, 3}, deep_ros::DataType::FLOAT32, allocator);
    REQUIRE_NOTHROW(node->run_inference(recovery_input));
  }

  SECTION("Lifecycle error recovery")
  {
    // Configure with valid plugin but problematic model
    auto params = std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("Backend.plugin", "mock_backend"), rclcpp::Parameter("model_path", "/problematic/model.onnx")};
    node->set_parameters(params);
    node->configure();
    node->activate();

    // Try to change to invalid model path
    auto bad_params = std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("model_path", "")  // Empty path should be rejected
    };
    auto result = node->set_parameters(bad_params);
    REQUIRE(result[0].successful == false);

    // Node should still be in good state
    REQUIRE(node->get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
    REQUIRE(node->is_plugin_loaded() == true);
    REQUIRE(node->is_model_loaded() == true);

    // Should be able to change to valid model
    auto good_params = std::vector<rclcpp::Parameter>{rclcpp::Parameter("model_path", "/valid/model.onnx")};
    auto result2 = node->set_parameters(good_params);
    REQUIRE(result2[0].successful == true);
  }
}

TEST_CASE_METHOD(deep_ros::test::TestExecutorFixture, "Integration: Resource Management", "[integration][resources]")
{
  SECTION("Multiple nodes can coexist")
  {
    // Create multiple nodes with different configurations
    rclcpp::NodeOptions options1;
    options1.parameter_overrides({{"Backend.plugin", "mock_backend"}});

    rclcpp::NodeOptions options2;
    options2.parameter_overrides({{"Backend.plugin", "mock_backend"}});

    auto node1 = std::make_shared<deep_ros::test::DeepTestNode>(options1);
    auto node2 = std::make_shared<deep_ros::test::DeepTestNode>(options2);

    add_node(node1);
    add_node(node2);
    start_spinning();

    // Both should configure independently
    node1->configure();
    node2->configure();

    REQUIRE(node1->is_plugin_loaded() == true);
    REQUIRE(node2->is_plugin_loaded() == true);

    // Both should have their own allocators
    auto alloc1 = node1->get_current_allocator();
    auto alloc2 = node2->get_current_allocator();
    REQUIRE(alloc1 != nullptr);
    REQUIRE(alloc2 != nullptr);
    // They might be same or different instances - both are valid
  }

  SECTION("Proper cleanup on node destruction")
  {
    // Create node in limited scope
    {
      auto node = std::make_shared<deep_ros::test::DeepTestNode>();
      add_node(node);
      start_spinning();

      auto params = std::vector<rclcpp::Parameter>{
        rclcpp::Parameter("Backend.plugin", "mock_backend"), rclcpp::Parameter("model_path", "/test/model.onnx")};
      node->set_parameters(params);
      node->configure();
      node->activate();

      REQUIRE(node->is_plugin_loaded() == true);
      REQUIRE(node->is_model_loaded() == true);

      // Explicit cleanup
      node->deactivate();
      node->cleanup();

      REQUIRE(node->is_plugin_loaded() == false);
      REQUIRE(node->is_model_loaded() == false);
    }
    // Node goes out of scope - should not crash
  }

  SECTION("Memory usage patterns")
  {
    auto node = std::make_shared<deep_ros::test::DeepTestNode>();
    add_node(node);
    start_spinning();

    auto params = std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("Backend.plugin", "mock_backend"), rclcpp::Parameter("model_path", "/test/model.onnx")};
    node->set_parameters(params);
    node->configure();
    node->activate();

    auto allocator = node->get_current_allocator();

    // Create and process many tensors
    for (int i = 0; i < 100; ++i) {
      std::vector<size_t> shape = {1, 3, 64, 64};  // ~49KB per tensor
      deep_ros::Tensor input(shape, deep_ros::DataType::UINT8, allocator);

      // Fill with data
      auto data = input.data_as<uint8_t>();
      for (size_t j = 0; j < input.size(); ++j) {
        data[j] = static_cast<uint8_t>(j % 256);
      }

      // Process
      auto output = node->run_inference(input);

      // Tensors should be automatically cleaned up
      REQUIRE(output.data() != nullptr);
    }

    // Node should still be functional after processing many tensors
    REQUIRE(node->is_plugin_loaded() == true);
    REQUIRE(node->is_model_loaded() == true);
  }
}
