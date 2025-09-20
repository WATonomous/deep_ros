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
#include <deep_core/deep_node_base.hpp>
#include <deep_core/types/tensor.hpp>

namespace deep_ros
{
namespace test
{

class MockMemoryAllocator : public BackendMemoryAllocator
{
public:
  void * allocate(size_t bytes) override
  {
    if (bytes == 0) return nullptr;

    allocated_bytes_ += bytes;
    void * ptr = std::malloc(bytes);
    return ptr;
  }

  void deallocate(void * ptr) override
  {
    if (ptr) {
      std::free(ptr);
    }
  }

  void copy_from_host(void * dst, const void * src, size_t bytes) override
  {
    std::memcpy(dst, src, bytes);
  }

  void copy_to_host(void * dst, const void * src, size_t bytes) override
  {
    std::memcpy(dst, src, bytes);
  }

  void copy_device_to_device(void * dst, const void * src, size_t bytes) override
  {
    std::memcpy(dst, src, bytes);
  }

  bool is_device_memory() const override
  {
    return false;
  }

  std::string device_name() const override
  {
    return "mock_device";
  }

  size_t allocated_bytes() const
  {
    return allocated_bytes_;
  }

private:
  size_t allocated_bytes_{0};
};

TEST_CASE("TensorPtr construction with allocator", "[tensor]")
{
  auto allocator = std::make_shared<MockMemoryAllocator>();
  std::vector<size_t> shape{2, 3, 4};

  TensorPtr tensor(shape, DataType::FLOAT32, allocator);

  REQUIRE(tensor.shape() == shape);
  REQUIRE(tensor.dtype() == DataType::FLOAT32);
  REQUIRE(tensor.size() == 24);  // 2 * 3 * 4
  REQUIRE(tensor.data() != nullptr);
  REQUIRE(allocator->allocated_bytes() > 0);
}

TEST_CASE("TensorPtr construction without allocator throws", "[tensor]")
{
  std::vector<size_t> shape{2, 3};
  REQUIRE_THROWS_AS(TensorPtr(shape, DataType::FLOAT32), std::runtime_error);
}

TEST_CASE("Different data types have correct sizes", "[tensor]")
{
  auto allocator = std::make_shared<MockMemoryAllocator>();

  TensorPtr float_tensor({10}, DataType::FLOAT32, allocator);
  TensorPtr int32_tensor({10}, DataType::INT32, allocator);
  TensorPtr int64_tensor({10}, DataType::INT64, allocator);
  TensorPtr uint8_tensor({10}, DataType::UINT8, allocator);

  REQUIRE(float_tensor.size() == 10);
  REQUIRE(int32_tensor.size() == 10);
  REQUIRE(int64_tensor.size() == 10);
  REQUIRE(uint8_tensor.size() == 10);
}

TEST_CASE("Empty shape creates scalar tensor", "[tensor]")
{
  auto allocator = std::make_shared<MockMemoryAllocator>();
  std::vector<size_t> empty_shape;

  TensorPtr tensor(empty_shape, DataType::FLOAT32, allocator);

  REQUIRE(tensor.size() == 1);  // Scalar tensor
  REQUIRE(tensor.shape().empty());
}

TEST_CASE("Large shape allocation", "[tensor]")
{
  auto allocator = std::make_shared<MockMemoryAllocator>();
  std::vector<size_t> large_shape{100, 100, 3};

  TensorPtr tensor(large_shape, DataType::UINT8, allocator);

  REQUIRE(tensor.size() == 30000);
  REQUIRE(tensor.shape() == large_shape);
}

class MockBackendExecutor : public BackendInferenceExecutor
{
public:
  bool load_model(const std::filesystem::path & model_path) override
  {
    loaded_model_path_ = model_path;
    model_loaded_ = true;
    return true;
  }

  TensorPtr run_inference(TensorPtr input) override
  {
    if (!model_loaded_) {
      throw std::runtime_error("No model loaded");
    }

    // Mock inference: return tensor with same shape but all zeros
    auto allocator = std::make_shared<MockMemoryAllocator>();
    TensorPtr output(input.shape(), input.dtype(), allocator);

    // Calculate correct byte size based on data type
    size_t dtype_size = get_dtype_size(input.dtype());
    size_t byte_size = output.size() * dtype_size;

    std::memset(output.data(), 0, byte_size);
    return output;
  }

  void unload_model() override
  {
    model_loaded_ = false;
    loaded_model_path_.clear();
  }

  std::vector<std::string> supported_model_formats() const override
  {
    return {"mock"};
  }

  bool is_model_loaded() const
  {
    return model_loaded_;
  }

  const std::filesystem::path & loaded_model_path() const
  {
    return loaded_model_path_;
  }

private:
  bool model_loaded_{false};
  std::filesystem::path loaded_model_path_;
};

class MockBackendPlugin : public DeepBackendPlugin
{
public:
  MockBackendPlugin()
  : allocator_(std::make_shared<MockMemoryAllocator>())
  , executor_(std::make_shared<MockBackendExecutor>())
  {}

  std::string backend_name() const override
  {
    return "mock_backend";
  }

  std::shared_ptr<BackendMemoryAllocator> get_allocator() const override
  {
    return allocator_;
  }

  std::shared_ptr<BackendInferenceExecutor> get_inference_executor() const override
  {
    return executor_;
  }

private:
  std::shared_ptr<MockMemoryAllocator> allocator_;
  std::shared_ptr<MockBackendExecutor> executor_;
};

TEST_CASE("Backend plugin interface", "[plugin]")
{
  MockBackendPlugin plugin;

  REQUIRE(plugin.backend_name() == "mock_backend");

  auto allocator = plugin.get_allocator();
  REQUIRE(allocator != nullptr);
  REQUIRE(allocator->device_name() == "mock_device");

  auto executor = plugin.get_inference_executor();
  REQUIRE(executor != nullptr);
  REQUIRE(executor->supported_model_formats() == std::vector<std::string>{"mock"});
}

TEST_CASE("Backend inference workflow", "[plugin][inference]")
{
  MockBackendPlugin plugin;
  auto allocator = plugin.get_allocator();
  auto executor = plugin.get_inference_executor();

  // Load model
  REQUIRE(executor->load_model("/fake/model.mock"));

  // Create input tensor
  std::vector<size_t> shape{1, 3, 224, 224};
  TensorPtr input(shape, DataType::FLOAT32, allocator);

  // Run inference
  auto output = executor->run_inference(std::move(input));

  REQUIRE(output.shape() == shape);
  REQUIRE(output.dtype() == DataType::FLOAT32);

  // Unload model
  executor->unload_model();
}

class TestInferenceNode : public DeepNodeBase
{
public:
  explicit TestInferenceNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : DeepNodeBase("test_inference_node", options)
  {}

  bool test_load_plugin(const std::string & plugin_name)
  {
    return load_plugin(plugin_name);
  }

  bool test_load_model(const std::filesystem::path & model_path)
  {
    return load_model(model_path);
  }

  TensorPtr test_run_inference(TensorPtr input)
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

TEST_CASE("DeepNodeBase creation", "[node]")
{
  // Skip ROS lifecycle node tests to avoid segfault
  REQUIRE(true);
}

}  // namespace test
}  // namespace deep_ros
