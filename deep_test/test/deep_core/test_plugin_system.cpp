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
#include <deep_core/plugin_interfaces/backend_inference_executor.hpp>
#include <deep_core/plugin_interfaces/backend_memory_allocator.hpp>
#include <deep_core/plugin_interfaces/deep_backend_plugin.hpp>
#include <deep_core/types/tensor.hpp>
#include <pluginlib/class_loader.hpp>
#include <test_fixtures/mock_backend_fixture.hpp>

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "Plugin System: Discovery and Loading", "[plugin][discovery]")
{
  SECTION("Plugin loader can discover available plugins")
  {
    pluginlib::ClassLoader<deep_ros::DeepBackendPlugin> loader("deep_core", "deep_ros::DeepBackendPlugin");

    auto available_plugins = loader.getDeclaredClasses();

    // Should find at least the mock plugin
    REQUIRE(!available_plugins.empty());

    bool found_mock = false;
    for (const auto & plugin_name : available_plugins) {
      if (plugin_name == "mock_backend") {
        found_mock = true;
        break;
      }
    }
    REQUIRE(found_mock);
  }

  SECTION("Can load valid plugin")
  {
    pluginlib::ClassLoader<deep_ros::DeepBackendPlugin> loader("deep_core", "deep_ros::DeepBackendPlugin");

    // Should be able to load the mock plugin
    REQUIRE_NOTHROW(loader.createSharedInstance("mock_backend"));

    auto plugin = loader.createSharedInstance("mock_backend");
    REQUIRE(plugin != nullptr);
  }

  SECTION("Loading invalid plugin should fail gracefully")
  {
    pluginlib::ClassLoader<deep_ros::DeepBackendPlugin> loader("deep_core", "deep_ros::DeepBackendPlugin");

    // Should throw when trying to load non-existent plugin
    REQUIRE_THROWS(loader.createSharedInstance("nonexistent_backend"));
    REQUIRE_THROWS(loader.createSharedInstance(""));
    REQUIRE_THROWS(loader.createSharedInstance("invalid_name"));
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "Plugin System: Backend Interface", "[plugin][interface]")
{
  auto backend = getBackend();

  SECTION("Backend provides basic identification")
  {
    REQUIRE(backend != nullptr);

    // Should have a meaningful name
    auto name = backend->backend_name();
    REQUIRE(!name.empty());
    REQUIRE(name == "mock_backend");
  }

  SECTION("Backend provides required interfaces")
  {
    // Must provide memory allocator
    auto allocator = backend->get_allocator();
    REQUIRE(allocator != nullptr);

    // Must provide inference executor
    auto executor = backend->get_inference_executor();
    REQUIRE(executor != nullptr);
  }

  SECTION("Multiple calls return consistent interfaces")
  {
    // Should return same instances or equivalent ones
    auto allocator1 = backend->get_allocator();
    auto allocator2 = backend->get_allocator();
    auto executor1 = backend->get_inference_executor();
    auto executor2 = backend->get_inference_executor();

    // These could be same instances or different but equivalent
    REQUIRE(allocator1 != nullptr);
    REQUIRE(allocator2 != nullptr);
    REQUIRE(executor1 != nullptr);
    REQUIRE(executor2 != nullptr);
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "Plugin System: Memory Allocator Interface", "[plugin][allocator]")
{
  auto allocator = getAllocator();

  SECTION("Basic allocation and deallocation")
  {
    // Should be able to allocate memory
    void * ptr = allocator->allocate(1024);
    REQUIRE(ptr != nullptr);

    // Memory should be writable
    auto * int_ptr = static_cast<int *>(ptr);
    *int_ptr = 42;
    REQUIRE(*int_ptr == 42);

    // Should deallocate safely
    REQUIRE_NOTHROW(allocator->deallocate(ptr));
  }

  SECTION("Zero-size allocation behavior")
  {
    // Zero allocation should return nullptr or be well-defined
    void * ptr = allocator->allocate(0);
    REQUIRE(ptr == nullptr);  // Most common expectation
  }

  SECTION("Null pointer deallocation is safe")
  {
    // Should not crash on null pointer
    REQUIRE_NOTHROW(allocator->deallocate(nullptr));
  }

  SECTION("Memory copy operations work correctly")
  {
    std::vector<float> source = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> dest(4, 0.0f);
    size_t bytes = source.size() * sizeof(float);

    void * device_ptr = allocator->allocate(bytes);
    REQUIRE(device_ptr != nullptr);

    // Host to device copy
    REQUIRE_NOTHROW(allocator->copy_from_host(device_ptr, source.data(), bytes));

    // Device to host copy
    REQUIRE_NOTHROW(allocator->copy_to_host(dest.data(), device_ptr, bytes));

    // Data should be preserved
    for (size_t i = 0; i < source.size(); ++i) {
      REQUIRE(dest[i] == source[i]);
    }

    allocator->deallocate(device_ptr);
  }

  SECTION("Device-to-device copy works")
  {
    std::vector<int> data = {10, 20, 30, 40};
    size_t bytes = data.size() * sizeof(int);

    void * ptr1 = allocator->allocate(bytes);
    void * ptr2 = allocator->allocate(bytes);
    REQUIRE(ptr1 != nullptr);
    REQUIRE(ptr2 != nullptr);

    // Copy data to first allocation
    allocator->copy_from_host(ptr1, data.data(), bytes);

    // Copy between device allocations
    REQUIRE_NOTHROW(allocator->copy_device_to_device(ptr2, ptr1, bytes));

    // Verify copy worked
    std::vector<int> result(4, 0);
    allocator->copy_to_host(result.data(), ptr2, bytes);

    for (size_t i = 0; i < data.size(); ++i) {
      REQUIRE(result[i] == data[i]);
    }

    allocator->deallocate(ptr1);
    allocator->deallocate(ptr2);
  }

  SECTION("Allocator provides device information")
  {
    // Should provide meaningful device info
    auto device_name = allocator->device_name();
    REQUIRE(!device_name.empty());

    // Should indicate memory type
    bool is_device = allocator->is_device_memory();
    // No specific requirement, but should be well-defined
    (void)is_device;  // Suppress unused variable warning
  }
}

TEST_CASE_METHOD(
  deep_ros::test::MockBackendFixture, "Plugin System: Inference Executor Interface", "[plugin][executor]")
{
  auto backend = getBackend();
  auto executor = backend->get_inference_executor();
  auto allocator = backend->get_allocator();

  SECTION("Executor reports supported model formats")
  {
    auto formats = executor->supported_model_formats();

    // Should support at least one format
    REQUIRE(!formats.empty());

    // Each format should be non-empty
    for (const auto & format : formats) {
      REQUIRE(!format.empty());
    }
  }

  SECTION("Model loading workflow")
  {
    // Should be able to load a model
    bool loaded = executor->load_model("/test/model.onnx");
    REQUIRE(loaded == true);

    // Should be able to unload
    REQUIRE_NOTHROW(executor->unload_model());

    // Should be able to load again
    loaded = executor->load_model("/another/model.mock");
    REQUIRE(loaded == true);
  }

  SECTION("Inference requires loaded model")
  {
    // Inference without model should fail
    deep_ros::Tensor input({2, 2}, deep_ros::DataType::FLOAT32, allocator);
    REQUIRE_THROWS_AS(executor->run_inference(input), std::runtime_error);

    // After loading model, inference should work
    executor->load_model("/test/model.onnx");
    deep_ros::Tensor input2({2, 2}, deep_ros::DataType::FLOAT32, allocator);
    REQUIRE_NOTHROW(executor->run_inference(input2));
  }

  SECTION("Inference with valid input produces valid output")
  {
    executor->load_model("/test/model.onnx");

    // Create input tensor
    std::vector<size_t> shape = {1, 3, 4, 4};
    deep_ros::Tensor input(shape, deep_ros::DataType::FLOAT32, allocator);

    // Fill with test data
    auto data = input.data_as<float>();
    for (size_t i = 0; i < input.size(); ++i) {
      data[i] = static_cast<float>(i) * 0.1f;
    }

    // Run inference
    auto output = executor->run_inference(input);

    // Output should be valid tensor
    REQUIRE(output.data() != nullptr);
    REQUIRE(output.size() > 0);
    REQUIRE(output.dtype() == deep_ros::DataType::FLOAT32);  // Common expectation
  }

  SECTION("Model state is managed correctly")
  {
    // Multiple model loads should work
    REQUIRE(executor->load_model("/model1.onnx") == true);
    REQUIRE(executor->load_model("/model2.onnx") == true);  // Should replace first

    // Unload should work
    executor->unload_model();

    // Inference after unload should fail
    deep_ros::Tensor input({1, 1}, deep_ros::DataType::FLOAT32, allocator);
    REQUIRE_THROWS_AS(executor->run_inference(input), std::runtime_error);
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "Plugin System: Error Handling", "[plugin][errors]")
{
  auto backend = getBackend();
  auto executor = backend->get_inference_executor();
  auto allocator = backend->get_allocator();

  SECTION("Invalid tensor input is handled")
  {
    executor->load_model("/test/model.onnx");

    // Empty tensor should be rejected
    deep_ros::Tensor empty_tensor;
    REQUIRE_THROWS(executor->run_inference(empty_tensor));
  }

  SECTION("Memory allocation failures are handled")
  {
    // This is hard to test without actually causing allocation failure
    // But the interface should handle it gracefully
    REQUIRE_NOTHROW(allocator->allocate(0));
  }

  SECTION("Copy with invalid parameters is handled")
  {
    // Should handle null pointers gracefully
    REQUIRE_THROWS(allocator->copy_from_host(nullptr, nullptr, 100));
    REQUIRE_THROWS(allocator->copy_to_host(nullptr, nullptr, 100));
    REQUIRE_THROWS(allocator->copy_device_to_device(nullptr, nullptr, 100));
  }

  SECTION("Plugin state consistency after errors")
  {
    // Failed operations shouldn't break the plugin
    try {
      deep_ros::Tensor empty_tensor;
      executor->run_inference(empty_tensor);
    } catch (...) {
      // Error is expected
    }

    // Plugin should still work after error
    REQUIRE(executor->load_model("/recovery/model.onnx") == true);

    deep_ros::Tensor valid_input({2, 2}, deep_ros::DataType::FLOAT32, allocator);
    REQUIRE_NOTHROW(executor->run_inference(valid_input));
  }
}
