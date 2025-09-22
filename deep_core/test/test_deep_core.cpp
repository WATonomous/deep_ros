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

#include <catch2/catch.hpp>
#include <pluginlib/class_loader.hpp>
#include <deep_core/deep_node_base.hpp>
#include <deep_core/types/tensor.hpp>
#include <deep_core/plugin_interfaces/deep_backend_plugin.hpp>
#include <deep_test/deep_test.hpp>

namespace deep_ros
{
namespace test
{

// Global test fixture for tensor tests using mock plugin
class TensorMockFixture
{
public:
  static TensorMockFixture & getInstance()
  {
    static TensorMockFixture instance;
    return instance;
  }

  std::shared_ptr<BackendMemoryAllocator> getAllocator()
  {
    return allocator_;
  }

private:
  TensorMockFixture()
  {
    loader_ = std::make_unique<pluginlib::ClassLoader<DeepBackendPlugin>>(
      "deep_core", "deep_ros::DeepBackendPlugin");
    mock_backend_ = loader_->createSharedInstance("mock_backend");
    allocator_ = mock_backend_->get_allocator();
  }

  std::unique_ptr<pluginlib::ClassLoader<DeepBackendPlugin>> loader_;
  std::shared_ptr<DeepBackendPlugin> mock_backend_;
  std::shared_ptr<BackendMemoryAllocator> allocator_;
};

TEST_CASE("Tensor construction with allocator", "[tensor]")
{
  auto allocator = TensorMockFixture::getInstance().getAllocator();
  std::vector<size_t> shape{2, 3, 4};

  Tensor tensor(shape, DataType::FLOAT32, allocator);

  REQUIRE(tensor.shape() == shape);
  REQUIRE(tensor.dtype() == DataType::FLOAT32);
  REQUIRE(tensor.size() == 24);  // 2 * 3 * 4
  REQUIRE(tensor.data() != nullptr);
}

TEST_CASE("Different data types have correct sizes", "[tensor]")
{
  auto allocator = TensorMockFixture::getInstance().getAllocator();

  Tensor float_tensor({10}, DataType::FLOAT32, allocator);
  Tensor int32_tensor({10}, DataType::INT32, allocator);
  Tensor int64_tensor({10}, DataType::INT64, allocator);
  Tensor uint8_tensor({10}, DataType::UINT8, allocator);

  REQUIRE(float_tensor.size() == 10);
  REQUIRE(int32_tensor.size() == 10);
  REQUIRE(int64_tensor.size() == 10);
  REQUIRE(uint8_tensor.size() == 10);
}

TEST_CASE("Empty shape throws exception", "[tensor]")
{
  auto allocator = TensorMockFixture::getInstance().getAllocator();
  std::vector<size_t> empty_shape;

  REQUIRE_THROWS_AS(Tensor(empty_shape, DataType::FLOAT32, allocator), std::invalid_argument);
}

TEST_CASE("Large shape allocation", "[tensor]")
{
  auto allocator = TensorMockFixture::getInstance().getAllocator();
  std::vector<size_t> large_shape{100, 100, 3};

  Tensor tensor(large_shape, DataType::UINT8, allocator);

  REQUIRE(tensor.size() == 30000);
  REQUIRE(tensor.shape() == large_shape);
}

// These tests will use the established mock_backend plugin through the node interface
// This provides more realistic testing of the actual plugin loading mechanism

class TestInferenceNode : public DeepNodeBase
{
public:
  explicit TestInferenceNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : DeepNodeBase("test_inference_node", options)
  {}

  // Expose protected methods for testing
  using DeepNodeBase::get_backend_name;
  using DeepNodeBase::get_current_allocator;
  using DeepNodeBase::is_model_loaded;
  using DeepNodeBase::is_plugin_loaded;
  using DeepNodeBase::load_model;
  using DeepNodeBase::load_plugin;
  using DeepNodeBase::run_inference;

  bool test_load_plugin(const std::string & plugin_name)
  {
    return load_plugin(plugin_name);
  }

  bool test_load_model(const std::filesystem::path & model_path)
  {
    return load_model(model_path);
  }

  Tensor test_run_inference(Tensor input)
  {
    return run_inference(input);
  }

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

TEST_CASE_METHOD(deep_ros::test::TestExecutorFixture, "DeepNodeBase lifecycle management", "[node]")
{
  // Create a test node that inherits from DeepNodeBase
  auto test_node = std::make_shared<TestInferenceNode>();
  add_node(test_node);
  start_spinning();

  SECTION("Node creation and initial state")
  {
    REQUIRE(test_node->get_name() == std::string("test_inference_node"));
    REQUIRE(test_node->get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
    REQUIRE(test_node->is_plugin_loaded() == false);
    REQUIRE(test_node->is_model_loaded() == false);
  }

  SECTION("Lifecycle transitions work correctly")
  {
    // Configure
    auto configure_result = test_node->configure();
    REQUIRE(configure_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

    // Activate
    auto activate_result = test_node->activate();
    REQUIRE(activate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

    // Deactivate
    auto deactivate_result = test_node->deactivate();
    REQUIRE(deactivate_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

    // Cleanup
    auto cleanup_result = test_node->cleanup();
    REQUIRE(cleanup_result.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  }

  SECTION("Plugin loading functionality")
  {
    // Configure first
    test_node->configure();

    // Try to load the test plugin
    bool load_result = test_node->test_load_plugin("mock_backend");
    REQUIRE(load_result == true);
    REQUIRE(test_node->is_plugin_loaded() == true);
    REQUIRE(test_node->get_backend_name() == "mock_backend");
    REQUIRE(test_node->get_current_allocator() != nullptr);

    // Try to load a non-existent plugin
    bool bad_load_result = test_node->test_load_plugin("nonexistent_plugin");
    REQUIRE(bad_load_result == false);
  }

  SECTION("Model loading and inference with plugin")
  {
    test_node->configure();

    // Load the test plugin
    bool plugin_result = test_node->test_load_plugin("mock_backend");
    REQUIRE(plugin_result == true);
    REQUIRE(test_node->is_plugin_loaded() == true);

    test_node->activate();

    // Load model with plugin
    bool model_result = test_node->test_load_model("/fake/model.onnx");
    REQUIRE(model_result == true);
    REQUIRE(test_node->is_model_loaded() == true);

    // Test inference
    auto allocator = test_node->get_current_allocator();
    REQUIRE(allocator != nullptr);

    std::vector<size_t> shape{1, 3, 224, 224};
    Tensor input(shape, DataType::FLOAT32, allocator);

    auto output = test_node->test_run_inference(std::move(input));
    REQUIRE(output.shape() == shape);
    REQUIRE(output.dtype() == DataType::FLOAT32);
  }

  SECTION("Model loading without plugin fails")
  {
    test_node->configure();
    test_node->activate();

    // Try to load model without plugin
    bool model_result = test_node->test_load_model("/fake/model.onnx");
    REQUIRE(model_result == false);
    REQUIRE(test_node->is_model_loaded() == false);

    // Verify backend state when no plugin is loaded
    REQUIRE(test_node->get_backend_name() == "none");
    REQUIRE(test_node->get_current_allocator() == nullptr);
  }
}

}  // namespace test
}  // namespace deep_ros
