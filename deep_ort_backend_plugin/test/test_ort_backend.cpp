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

#include <catch2/catch.hpp>

#include <deep_ort_backend_plugin/ort_backend_plugin.hpp>
#include <deep_ort_backend_plugin/ort_cpu_memory_allocator.hpp>
#include <deep_ort_backend_plugin/ort_backend_executor.hpp>

#include <filesystem>
#include <chrono>
#include <thread>

namespace deep_ort_backend
{
namespace test
{

TEST_CASE("OrtCpuMemoryAllocator basic allocation", "[allocator]")
{
  OrtCpuMemoryAllocator allocator;
  
  SECTION("Basic allocation and deallocation") {
    void* ptr = allocator.allocate(1024);
    REQUIRE(ptr != nullptr);

    // Test that memory is properly aligned (64-byte alignment)
    REQUIRE(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);

    allocator.deallocate(ptr);
  }
  
  SECTION("Zero allocation returns nullptr") {
    void* ptr = allocator.allocate(0);
    REQUIRE(ptr == nullptr);
    
    allocator.deallocate(ptr);  // Should not crash
  }
  
  SECTION("Large allocation") {
    const size_t large_size = 1024 * 1024;  // 1MB
    void* ptr = allocator.allocate(large_size);
    REQUIRE(ptr != nullptr);
    
    // Verify alignment on large allocations
    REQUIRE(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
    
    allocator.deallocate(ptr);
  }
}

TEST_CASE("OrtCpuMemoryAllocator memory operations", "[allocator]")
{
  OrtCpuMemoryAllocator allocator;
  
  SECTION("Memory copy operations") {
    const size_t size = 256;
    std::vector<uint8_t> src_data(size, 0x42);
    std::vector<uint8_t> dst_data(size, 0x00);

    allocator.copy_from_host(dst_data.data(), src_data.data(), size);
    
    REQUIRE(src_data == dst_data);
  }
  
  SECTION("Device properties") {
    REQUIRE_FALSE(allocator.is_device_memory());
    REQUIRE(allocator.device_name() == "ort_cpu");
  }
  
  SECTION("Copy device to device") {
    const size_t size = 128;
    std::vector<uint8_t> src(size, 0xAB);
    std::vector<uint8_t> dst(size, 0x00);
    
    allocator.copy_device_to_device(dst.data(), src.data(), size);
    
    REQUIRE(src == dst);
  }
}

TEST_CASE("OrtBackendExecutor basic functionality", "[executor]")
{
  OrtBackendExecutor executor;
  auto allocator = get_ort_cpu_allocator();
  
  SECTION("Supported formats") {
    auto formats = executor.supported_model_formats();
    REQUIRE(formats.size() == 1);
    REQUIRE(formats[0] == "onnx");
  }
  
  SECTION("Load nonexistent model fails") {
    bool result = executor.load_model("/nonexistent/model.onnx");
    REQUIRE_FALSE(result);
  }
  
  SECTION("Inference without model throws") {
    std::vector<size_t> shape{1, 3, 224, 224};
    deep_ros::Tensor input(shape, deep_ros::DataType::FLOAT32, allocator);
    
    REQUIRE_THROWS_AS(executor.run_inference(input), std::runtime_error);
  }
  
  SECTION("Unload model doesn't crash") {
    executor.unload_model();
    // Should not crash
  }
}

TEST_CASE("OrtBackendExecutor data type handling", "[executor][datatypes]")
{
  auto allocator = get_ort_cpu_allocator();
  std::vector<size_t> shape{2, 2};
  
  SECTION("FLOAT32 tensor creation") {
    deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, allocator);
    REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);
  }
  
  SECTION("INT32 tensor creation") {
    deep_ros::Tensor tensor(shape, deep_ros::DataType::INT32, allocator);
    REQUIRE(tensor.dtype() == deep_ros::DataType::INT32);
  }
  
  SECTION("INT64 tensor creation") {
    deep_ros::Tensor tensor(shape, deep_ros::DataType::INT64, allocator);
    REQUIRE(tensor.dtype() == deep_ros::DataType::INT64);
  }
  
  SECTION("UINT8 tensor creation") {
    deep_ros::Tensor tensor(shape, deep_ros::DataType::UINT8, allocator);
    REQUIRE(tensor.dtype() == deep_ros::DataType::UINT8);
  }
}

TEST_CASE("OrtBackendPlugin interface", "[plugin]")
{
  OrtBackendPlugin plugin;
  
  SECTION("Basic plugin properties") {
    REQUIRE(plugin.backend_name() == "onnxruntime_cpu");
    
    auto allocator = plugin.get_allocator();
    REQUIRE(allocator != nullptr);
    
    auto executor = plugin.get_inference_executor();
    REQUIRE(executor != nullptr);
  }
  
  SECTION("Allocator integration") {
    auto allocator = plugin.get_allocator();
    
    REQUIRE_FALSE(allocator->is_device_memory());
    REQUIRE(allocator->device_name() == "ort_cpu");
    
    // Test allocation through plugin interface
    void* ptr = allocator->allocate(512);
    REQUIRE(ptr != nullptr);
    REQUIRE(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
    
    allocator->deallocate(ptr);
  }
  
  SECTION("Executor integration") {
    auto executor = plugin.get_inference_executor();
    
    auto formats = executor->supported_model_formats();
    REQUIRE(formats == std::vector<std::string>{"onnx"});
    
    // Test that unload doesn't crash
    executor->unload_model();
  }
}

TEST_CASE("Global allocator singleton behavior", "[allocator][singleton]")
{
  auto allocator1 = get_ort_cpu_allocator();
  auto allocator2 = get_ort_cpu_allocator();
  
  SECTION("Returns same instance") {
    REQUIRE(allocator1.get() == allocator2.get());
    
    // Both should have the same properties
    REQUIRE(allocator1->device_name() == allocator2->device_name());
    REQUIRE(allocator1->is_device_memory() == allocator2->is_device_memory());
  }
  
  SECTION("Thread safety test") {
    std::vector<std::thread> threads;
    std::vector<std::shared_ptr<deep_ros::BackendMemoryAllocator>> allocators(10);
    
    for (int i = 0; i < 10; ++i) {
      threads.emplace_back([&allocators, i]() {
        allocators[i] = get_ort_cpu_allocator();
      });
    }
    
    for (auto& thread : threads) {
      thread.join();
    }
    
    // All allocators should be the same instance
    for (const auto& allocator : allocators) {
      REQUIRE(allocator.get() == allocator1.get());
    }
  }
}

TEST_CASE("Integration test - full workflow", "[integration]")
{
  OrtBackendPlugin plugin;
  auto allocator = plugin.get_allocator();
  auto executor = plugin.get_inference_executor();
  
  SECTION("Tensor creation with ORT allocator") {
    std::vector<size_t> shape{2, 3, 4};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, allocator);
    
    REQUIRE(tensor.shape() == shape);
    REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);
    REQUIRE(tensor.size() == 24);
    REQUIRE(tensor.data() != nullptr);
    
    // Verify memory is aligned
    REQUIRE(reinterpret_cast<uintptr_t>(tensor.data()) % 64 == 0);
  }
  
  SECTION("Multiple allocations") {
    std::vector<deep_ros::Tensor> tensors;
    
    for (int i = 0; i < 5; ++i) {
      std::vector<size_t> shape{static_cast<size_t>(i + 1), 10};
      tensors.emplace_back(shape, deep_ros::DataType::FLOAT32, allocator);
    }
    
    // Verify all tensors are properly allocated
    for (size_t i = 0; i < tensors.size(); ++i) {
      REQUIRE(tensors[i].size() == (i + 1) * 10);
      REQUIRE(tensors[i].data() != nullptr);
    }
  }
}

TEST_CASE("Performance characteristics", "[performance]")
{
  auto allocator = get_ort_cpu_allocator();
  
  SECTION("Allocation speed") {
    const std::vector<size_t> sizes = {1024, 4096, 16384, 65536};
    
    for (size_t size : sizes) {
      auto start = std::chrono::high_resolution_clock::now();
      
      void* ptr = allocator->allocate(size);
      REQUIRE(ptr != nullptr);
      
      auto end = std::chrono::high_resolution_clock::now();
      allocator->deallocate(ptr);
      
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      
      // Allocation should complete within reasonable time (1ms for large allocations)
      REQUIRE(duration.count() < 1000);
    }
  }
  
  SECTION("Memory alignment verification") {
    const std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048, 4096};
    
    for (size_t size : sizes) {
      void* ptr = allocator->allocate(size);
      REQUIRE(ptr != nullptr);
      REQUIRE(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
      allocator->deallocate(ptr);
    }
  }
}

TEST_CASE("Error handling", "[error-handling]")
{
  OrtBackendExecutor executor;
  auto allocator = get_ort_cpu_allocator();
  
  SECTION("Inference without loaded model") {
    std::vector<size_t> shape{1, 1};
    deep_ros::Tensor input(shape, deep_ros::DataType::FLOAT32, allocator);
    
    REQUIRE_THROWS_WITH(
      executor.run_inference(input),
      "No model loaded for inference"
    );
  }
  
  SECTION("Null pointer handling in allocator") {
    OrtCpuMemoryAllocator allocator;
    
    // Deallocating nullptr should not crash
    allocator.deallocate(nullptr);
    
    // Copy operations with null should not crash (though undefined behavior)
    allocator.copy_from_host(nullptr, nullptr, 0);
    allocator.copy_to_host(nullptr, nullptr, 0);
    allocator.copy_device_to_device(nullptr, nullptr, 0);
  }
}

}  // namespace test
}  // namespace deep_ort_backend