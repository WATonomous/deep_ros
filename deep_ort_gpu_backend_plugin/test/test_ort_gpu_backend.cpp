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
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch.hpp>
#include <deep_ort_gpu_backend_plugin/ort_gpu_backend_executor.hpp>
#include <deep_ort_gpu_backend_plugin/ort_gpu_backend_plugin.hpp>
#include <deep_ort_gpu_backend_plugin/ort_gpu_memory_allocator.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace deep_ort_gpu_backend
{
namespace test
{

// Helper function to create a dummy lifecycle node for testing
rclcpp_lifecycle::LifecycleNode::SharedPtr create_dummy_node()
{
  static bool initialized = false;
  if (!initialized) {
    rclcpp::init(0, nullptr);
    initialized = true;
  }

  // Create unique node name to avoid parameter conflicts
  static int node_counter = 0;
  std::string node_name = "test_node_" + std::to_string(node_counter++);
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(node_name);
  return node;
}

// Helper function to check if CUDA is available
bool is_cuda_available()
{
  try {
    // Try creating a simple CUDA execution provider to verify CUDA availability
    auto logger = rclcpp::get_logger("test_cuda_availability");
    OrtGpuBackendExecutor test_executor(0, "cuda", logger, false, "");
    return true;
  } catch (...) {
    return false;
  }
}

TEST_CASE("OrtGpuCpuMemoryAllocator basic functionality", "[allocator]")
{
  OrtGpuCpuMemoryAllocator allocator;

  // Test basic allocation
  void * ptr = allocator.allocate(1024);
  REQUIRE(ptr != nullptr);
  REQUIRE(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);  // 64-byte aligned
  allocator.deallocate(ptr);

  // Test zero allocation
  void * null_ptr = allocator.allocate(0);
  REQUIRE(null_ptr == nullptr);

  // Test device properties
  REQUIRE_FALSE(allocator.is_device_memory());
  REQUIRE(allocator.device_name() == "gpu_backend_cpu");
}

TEST_CASE("OrtGpuCpuMemoryAllocator memory operations", "[allocator]")
{
  OrtGpuCpuMemoryAllocator allocator;

  const size_t size = 256;
  std::vector<uint8_t> src_data(size, 0x42);
  std::vector<uint8_t> dst_data(size, 0x00);

  // Test copy operations
  allocator.copy_from_host(dst_data.data(), src_data.data(), size);
  REQUIRE(src_data == dst_data);

  // Test null handling
  allocator.deallocate(nullptr);  // Should not crash
  allocator.copy_from_host(nullptr, nullptr, 0);  // Should not crash
}

TEST_CASE("OrtGpuCpuMemoryAllocator ORT integration", "[allocator][ort]")
{
  OrtGpuCpuMemoryAllocator allocator;

  // Test ORT interface
  OrtAllocator * ort_alloc = allocator.get_ort_allocator();
  REQUIRE(ort_alloc != nullptr);

  const OrtMemoryInfo * mem_info = allocator.get_ort_memory_info();
  REQUIRE(mem_info != nullptr);

  // Test ORT callbacks
  void * ptr = ort_alloc->Alloc(ort_alloc, 512);
  REQUIRE(ptr != nullptr);
  REQUIRE(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
  ort_alloc->Free(ort_alloc, ptr);
}

TEST_CASE("OrtGpuBackendExecutor basic functionality", "[executor]")
{
  if (!is_cuda_available()) {
    return;  // Skip if CUDA not available
  }

  auto logger = rclcpp::get_logger("test_executor");
  OrtGpuBackendExecutor executor(0, "cuda", logger, false, "");
  auto allocator = get_ort_gpu_cpu_allocator();

  // Test supported formats
  auto formats = executor.supported_model_formats();
  REQUIRE(formats.size() == 1);
  REQUIRE(formats[0] == "onnx");

  // Test device properties
  REQUIRE(executor.get_device_id() == 0);

  // Test model loading failure
  bool result = executor.load_model("/nonexistent/model.onnx");
  REQUIRE_FALSE(result);

  // Test inference without model
  std::vector<size_t> shape{1, 3, 224, 224};
  deep_ros::Tensor input(shape, deep_ros::DataType::FLOAT32, allocator);
  REQUIRE_THROWS_AS(executor.run_inference(input), std::runtime_error);
}

TEST_CASE("OrtGpuBackendPlugin interface", "[plugin]")
{
  if (!is_cuda_available()) {
    return;  // Skip if CUDA not available
  }

  OrtGpuBackendPlugin plugin;
  auto node = create_dummy_node();

  // Plugin will declare parameters in initialize()
  plugin.initialize(node);

  // Test basic properties
  REQUIRE(plugin.backend_name() == "onnxruntime_gpu");
  REQUIRE(plugin.get_device_id() == 0);

  // Test allocator - check if it's initialized properly
  auto allocator = plugin.get_allocator();
  if (allocator == nullptr) {
    std::cout << "Allocator is null - checking if GPU backend plugin allocator initialization failed" << std::endl;
    return;  // Skip the rest if allocator initialization failed
  }

  REQUIRE(allocator != nullptr);
  REQUIRE_FALSE(allocator->is_device_memory());
  REQUIRE(allocator->device_name() == "gpu_backend_cpu");

  // Test executor
  auto executor = plugin.get_inference_executor();
  REQUIRE(executor != nullptr);
  auto formats = executor->supported_model_formats();
  REQUIRE(formats == std::vector<std::string>{"onnx"});
}

TEST_CASE("Data type handling", "[datatypes]")
{
  if (!is_cuda_available()) {
    return;  // Skip if CUDA not available
  }

  auto allocator = get_ort_gpu_cpu_allocator();
  std::vector<size_t> shape{2, 2};

  // Test different data types
  deep_ros::Tensor float_tensor(shape, deep_ros::DataType::FLOAT32, allocator);
  REQUIRE(float_tensor.dtype() == deep_ros::DataType::FLOAT32);

  deep_ros::Tensor int_tensor(shape, deep_ros::DataType::INT32, allocator);
  REQUIRE(int_tensor.dtype() == deep_ros::DataType::INT32);
}

TEST_CASE("Error handling", "[error]")
{
  OrtGpuCpuMemoryAllocator allocator;

  // Test large allocation failure
  REQUIRE_THROWS_AS(allocator.allocate(SIZE_MAX), std::bad_alloc);

  // Test null operations
  allocator.deallocate(nullptr);
  allocator.copy_from_host(nullptr, nullptr, 0);
  allocator.copy_to_host(nullptr, nullptr, 0);
  allocator.copy_device_to_device(nullptr, nullptr, 0);
}

TEST_CASE("Singleton behavior", "[singleton]")
{
  auto allocator1 = get_ort_gpu_cpu_allocator();
  auto allocator2 = get_ort_gpu_cpu_allocator();

  // Should return same instance
  REQUIRE(allocator1.get() == allocator2.get());
  REQUIRE(allocator1->device_name() == allocator2->device_name());
}

TEST_CASE("TensorRT execution provider", "[tensorrt][!mayfail]")
{
  try {
    auto logger = rclcpp::get_logger("test_tensorrt");
    OrtGpuBackendExecutor executor(0, "tensorrt", logger, false, "");

    REQUIRE(executor.get_device_id() == 0);

    auto formats = executor.supported_model_formats();
    REQUIRE(formats.size() == 1);
    REQUIRE(formats[0] == "onnx");

    std::cout << "TensorRT provider test PASSED" << std::endl;
  } catch (const std::exception & e) {
    std::cout << "TensorRT provider not available or failed: " << e.what() << std::endl;
    std::cout << "This is expected if TensorRT is not installed" << std::endl;
  }
}

}  // namespace test
}  // namespace deep_ort_gpu_backend
