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

#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch.hpp>
#include <deep_ort_gpu_backend_plugin/ort_gpu_backend_executor.hpp>
#include <deep_ort_gpu_backend_plugin/ort_gpu_backend_plugin.hpp>
#include <deep_ort_gpu_backend_plugin/ort_gpu_memory_allocator.hpp>

namespace deep_ort_gpu_backend
{
namespace test
{

// Helper function to check if CUDA is available
bool is_cuda_available()
{
  int device_count = 0;
  cudaError_t result = cudaGetDeviceCount(&device_count);
  return result == cudaSuccess && device_count > 0;
}

TEST_CASE("OrtGpuMemoryAllocator basic allocation", "[allocator][gpu]")
{
  if (!is_cuda_available()) {
    return;
  }

  OrtGpuMemoryAllocator allocator(0);

  SECTION("Basic allocation and deallocation")
  {
    void * ptr = allocator.allocate(1024);
    REQUIRE(ptr != nullptr);

    // Verify this is GPU memory
    cudaPointerAttributes attributes;
    cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);
    REQUIRE(result == cudaSuccess);
    REQUIRE(attributes.type == cudaMemoryTypeDevice);

    allocator.deallocate(ptr);
  }

  SECTION("Zero allocation returns nullptr")
  {
    void * ptr = allocator.allocate(0);
    REQUIRE(ptr == nullptr);

    allocator.deallocate(ptr);  // Should not crash
  }

  SECTION("Large allocation")
  {
    const size_t large_size = 1024 * 1024;  // 1MB
    void * ptr = allocator.allocate(large_size);
    REQUIRE(ptr != nullptr);

    // Verify this is GPU memory
    cudaPointerAttributes attributes;
    cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);
    REQUIRE(result == cudaSuccess);
    REQUIRE(attributes.type == cudaMemoryTypeDevice);

    allocator.deallocate(ptr);
  }

  SECTION("Multiple device support")
  {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count > 1) {
      OrtGpuMemoryAllocator allocator_device1(1);
      REQUIRE(allocator_device1.get_device_id() == 1);
      REQUIRE(allocator_device1.device_name() == "cuda:1");
    }
  }
}

TEST_CASE("OrtGpuMemoryAllocator memory operations", "[allocator][gpu]")
{
  if (!is_cuda_available()) {
    return;
  }

  OrtGpuMemoryAllocator allocator(0);

  SECTION("Host to device copy")
  {
    const size_t size = 256;
    std::vector<uint8_t> host_data(size, 0x42);
    std::vector<uint8_t> result_data(size, 0x00);

    void * device_ptr = allocator.allocate(size);
    REQUIRE(device_ptr != nullptr);

    // Copy from host to device
    allocator.copy_from_host(device_ptr, host_data.data(), size);

    // Copy back to verify
    allocator.copy_to_host(result_data.data(), device_ptr, size);

    REQUIRE(host_data == result_data);

    allocator.deallocate(device_ptr);
  }

  SECTION("Device to device copy")
  {
    const size_t size = 128;
    std::vector<uint8_t> host_data(size, 0xAB);
    std::vector<uint8_t> result_data(size, 0x00);

    void * device_ptr1 = allocator.allocate(size);
    void * device_ptr2 = allocator.allocate(size);
    REQUIRE(device_ptr1 != nullptr);
    REQUIRE(device_ptr2 != nullptr);

    // Upload data to first device buffer
    allocator.copy_from_host(device_ptr1, host_data.data(), size);

    // Copy between device buffers
    allocator.copy_device_to_device(device_ptr2, device_ptr1, size);

    // Download and verify
    allocator.copy_to_host(result_data.data(), device_ptr2, size);

    REQUIRE(host_data == result_data);

    allocator.deallocate(device_ptr1);
    allocator.deallocate(device_ptr2);
  }

  SECTION("Device properties")
  {
    REQUIRE(allocator.is_device_memory());
    REQUIRE(allocator.device_name() == "cuda:0");
    REQUIRE(allocator.get_device_id() == 0);
  }
}

TEST_CASE("OrtGpuBackendExecutor basic functionality", "[executor][gpu]")
{
  if (!is_cuda_available()) {
    return;
  }

  OrtGpuBackendExecutor executor(0, GpuExecutionProvider::CUDA);
  auto allocator = get_ort_gpu_allocator(0);

  SECTION("Supported formats")
  {
    auto formats = executor.supported_model_formats();
    REQUIRE(formats.size() == 1);
    REQUIRE(formats[0] == "onnx");
  }

  SECTION("Device properties")
  {
    REQUIRE(executor.get_device_id() == 0);
    REQUIRE(executor.get_execution_provider() == GpuExecutionProvider::CUDA);
  }

  SECTION("Load nonexistent model fails")
  {
    bool result = executor.load_model("/nonexistent/model.onnx");
    REQUIRE_FALSE(result);
  }

  SECTION("Inference without model throws")
  {
    std::vector<size_t> shape{1, 3, 224, 224};
    deep_ros::Tensor input(shape, deep_ros::DataType::FLOAT32, allocator);

    REQUIRE_THROWS_AS(executor.run_inference(input), std::runtime_error);
  }

  SECTION("Unload model doesn't crash")
  {
    executor.unload_model();
    // Should not crash
  }
}

TEST_CASE("OrtGpuBackendExecutor execution providers", "[executor][gpu][providers]")
{
  if (!is_cuda_available()) {
    return;
  }

  SECTION("CUDA execution provider")
  {
    OrtGpuBackendExecutor cuda_executor(0, GpuExecutionProvider::CUDA);
    REQUIRE(cuda_executor.get_execution_provider() == GpuExecutionProvider::CUDA);
  }

  SECTION("TensorRT execution provider")
  {
    // TensorRT might not be available, so we test construction but not necessarily inference
    REQUIRE_NOTHROW([&]() {
      OrtGpuBackendExecutor tensorrt_executor(0, GpuExecutionProvider::TENSORRT);
      REQUIRE(tensorrt_executor.get_execution_provider() == GpuExecutionProvider::TENSORRT);
    }());
  }
}

TEST_CASE("OrtGpuBackendExecutor data type handling", "[executor][gpu][datatypes]")
{
  if (!is_cuda_available()) {
    return;
  }

  auto allocator = get_ort_gpu_allocator(0);
  std::vector<size_t> shape{2, 2};

  SECTION("FLOAT32 tensor creation")
  {
    deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, allocator);
    REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);
  }

  SECTION("INT32 tensor creation")
  {
    deep_ros::Tensor tensor(shape, deep_ros::DataType::INT32, allocator);
    REQUIRE(tensor.dtype() == deep_ros::DataType::INT32);
  }

  SECTION("INT64 tensor creation")
  {
    deep_ros::Tensor tensor(shape, deep_ros::DataType::INT64, allocator);
    REQUIRE(tensor.dtype() == deep_ros::DataType::INT64);
  }

  SECTION("UINT8 tensor creation")
  {
    deep_ros::Tensor tensor(shape, deep_ros::DataType::UINT8, allocator);
    REQUIRE(tensor.dtype() == deep_ros::DataType::UINT8);
  }

  // SECTION("FLOAT16 tensor creation")  // Not available in current deep_ros version
  // {
  //   deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT16, allocator);
  //   REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT16);
  // }
}

TEST_CASE("OrtGpuBackendPlugin interface", "[plugin][gpu]")
{
  if (!is_cuda_available()) {
    return;
  }

  OrtGpuBackendPlugin plugin(0, GpuExecutionProvider::CUDA);

  SECTION("Basic plugin properties")
  {
    REQUIRE(plugin.backend_name() == "onnxruntime_gpu");
    REQUIRE(plugin.get_device_id() == 0);
    REQUIRE(plugin.get_execution_provider() == GpuExecutionProvider::CUDA);

    auto allocator = plugin.get_allocator();
    REQUIRE(allocator != nullptr);

    auto executor = plugin.get_inference_executor();
    REQUIRE(executor != nullptr);
  }

  SECTION("Allocator integration")
  {
    auto allocator = plugin.get_allocator();

    REQUIRE(allocator->is_device_memory());
    REQUIRE(allocator->device_name() == "cuda:0");

    // Test allocation through plugin interface
    void * ptr = allocator->allocate(512);
    REQUIRE(ptr != nullptr);

    // Verify this is GPU memory
    cudaPointerAttributes attributes;
    cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);
    REQUIRE(result == cudaSuccess);
    REQUIRE(attributes.type == cudaMemoryTypeDevice);

    allocator->deallocate(ptr);
  }

  SECTION("Executor integration")
  {
    auto executor = plugin.get_inference_executor();

    auto formats = executor->supported_model_formats();
    REQUIRE(formats == std::vector<std::string>{"onnx"});

    // Test that unload doesn't crash
    executor->unload_model();
  }
}

TEST_CASE("Multi-device GPU support", "[allocator][gpu][multidevice]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  if (device_count < 2) {
    return;  // Skip test - Multiple CUDA devices not available
  }

  SECTION("Different device allocators")
  {
    auto allocator0 = get_ort_gpu_allocator(0);
    auto allocator1 = get_ort_gpu_allocator(1);

    REQUIRE(allocator0->device_name() == "cuda:0");
    REQUIRE(allocator1->device_name() == "cuda:1");

    // Allocators for different devices should be different instances
    REQUIRE(allocator0.get() != allocator1.get());
  }

  SECTION("Cross-device memory operations")
  {
    OrtGpuMemoryAllocator allocator0(0);
    OrtGpuMemoryAllocator allocator1(1);

    const size_t size = 256;
    std::vector<uint8_t> host_data(size, 0x55);
    std::vector<uint8_t> result_data(size, 0x00);

    void * ptr0 = allocator0.allocate(size);
    void * ptr1 = allocator1.allocate(size);

    REQUIRE(ptr0 != nullptr);
    REQUIRE(ptr1 != nullptr);

    // Upload to device 0
    allocator0.copy_from_host(ptr0, host_data.data(), size);

    // Note: Direct device-to-device copy between different devices
    // would require peer-to-peer access setup, which is complex
    // So we test through host memory
    allocator0.copy_to_host(result_data.data(), ptr0, size);
    allocator1.copy_from_host(ptr1, result_data.data(), size);

    // Verify data integrity
    std::fill(result_data.begin(), result_data.end(), 0x00);
    allocator1.copy_to_host(result_data.data(), ptr1, size);

    REQUIRE(host_data == result_data);

    allocator0.deallocate(ptr0);
    allocator1.deallocate(ptr1);
  }
}

TEST_CASE("GPU allocator singleton behavior", "[allocator][gpu][singleton]")
{
  if (!is_cuda_available()) {
    return;
  }

  auto allocator1 = get_ort_gpu_allocator(0);
  auto allocator2 = get_ort_gpu_allocator(0);

  SECTION("Returns same instance for same device")
  {
    REQUIRE(allocator1.get() == allocator2.get());

    // Both should have the same properties
    REQUIRE(allocator1->device_name() == allocator2->device_name());
    REQUIRE(allocator1->is_device_memory() == allocator2->is_device_memory());
  }

  SECTION("Thread safety test")
  {
    std::vector<std::thread> threads;
    std::vector<std::shared_ptr<deep_ros::BackendMemoryAllocator>> allocators(10);

    for (int i = 0; i < 10; ++i) {
      threads.emplace_back([&allocators, i]() { allocators[i] = get_ort_gpu_allocator(0); });
    }

    for (auto & thread : threads) {
      thread.join();
    }

    // All allocators should be the same instance
    for (const auto & allocator : allocators) {
      REQUIRE(allocator.get() == allocator1.get());
    }
  }
}

TEST_CASE("Integration test - full GPU workflow", "[integration][gpu]")
{
  if (!is_cuda_available()) {
    return;
  }

  OrtGpuBackendPlugin plugin(0, GpuExecutionProvider::CUDA);
  auto allocator = plugin.get_allocator();
  auto executor = plugin.get_inference_executor();

  SECTION("GPU tensor creation with ORT allocator")
  {
    std::vector<size_t> shape{2, 3, 4};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, allocator);

    REQUIRE(tensor.shape() == shape);
    REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);
    REQUIRE(tensor.size() == 24);
    REQUIRE(tensor.data() != nullptr);

    // Verify memory is on GPU
    cudaPointerAttributes attributes;
    cudaError_t result = cudaPointerGetAttributes(&attributes, tensor.data());
    REQUIRE(result == cudaSuccess);
    REQUIRE(attributes.type == cudaMemoryTypeDevice);
  }

  SECTION("Multiple GPU allocations")
  {
    std::vector<deep_ros::Tensor> tensors;

    for (int i = 0; i < 5; ++i) {
      std::vector<size_t> shape{static_cast<size_t>(i + 1), 10};
      tensors.emplace_back(shape, deep_ros::DataType::FLOAT32, allocator);
    }

    // Verify all tensors are properly allocated on GPU
    for (size_t i = 0; i < tensors.size(); ++i) {
      REQUIRE(tensors[i].size() == (i + 1) * 10);
      REQUIRE(tensors[i].data() != nullptr);

      cudaPointerAttributes attributes;
      cudaError_t result = cudaPointerGetAttributes(&attributes, tensors[i].data());
      REQUIRE(result == cudaSuccess);
      REQUIRE(attributes.type == cudaMemoryTypeDevice);
    }
  }
}

TEST_CASE("GPU performance characteristics", "[performance][gpu]")
{
  if (!is_cuda_available()) {
    return;
  }

  auto allocator = get_ort_gpu_allocator(0);

  SECTION("GPU allocation speed")
  {
    const std::vector<size_t> sizes = {1024, 4096, 16384, 65536};

    for (size_t size : sizes) {
      auto start = std::chrono::high_resolution_clock::now();

      void * ptr = allocator->allocate(size);
      REQUIRE(ptr != nullptr);

      auto end = std::chrono::high_resolution_clock::now();
      allocator->deallocate(ptr);

      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      // GPU allocation should complete within reasonable time (10ms for large allocations)
      REQUIRE(duration.count() < 10000);
    }
  }

  SECTION("Memory transfer benchmarks")
  {
    const size_t size = 1024 * 1024;  // 1MB
    std::vector<uint8_t> host_data(size, 0xFF);
    std::vector<uint8_t> result_data(size, 0x00);

    void * device_ptr = allocator->allocate(size);
    REQUIRE(device_ptr != nullptr);

    // Benchmark host-to-device transfer
    auto start = std::chrono::high_resolution_clock::now();
    allocator->copy_from_host(device_ptr, host_data.data(), size);
    cudaDeviceSynchronize();  // Ensure transfer is complete
    auto mid = std::chrono::high_resolution_clock::now();

    // Benchmark device-to-host transfer
    allocator->copy_to_host(result_data.data(), device_ptr, size);
    cudaDeviceSynchronize();  // Ensure transfer is complete
    auto end = std::chrono::high_resolution_clock::now();

    auto h2d_duration = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto d2h_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);

    // Memory transfers should complete within reasonable time (100ms for 1MB)
    REQUIRE(h2d_duration.count() < 100000);
    REQUIRE(d2h_duration.count() < 100000);

    // Verify data integrity
    REQUIRE(host_data == result_data);

    allocator->deallocate(device_ptr);
  }
}

TEST_CASE("GPU error handling", "[error-handling][gpu]")
{
  if (!is_cuda_available()) {
    return;
  }

  SECTION("Invalid device ID")
  {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    // Try to create allocator with invalid device ID
    REQUIRE_THROWS_AS(OrtGpuMemoryAllocator(device_count + 10), std::runtime_error);
  }

  SECTION("Inference without loaded model")
  {
    OrtGpuBackendExecutor executor(0, GpuExecutionProvider::CUDA);
    auto allocator = get_ort_gpu_allocator(0);

    std::vector<size_t> shape{1, 1};
    deep_ros::Tensor input(shape, deep_ros::DataType::FLOAT32, allocator);

    REQUIRE_THROWS_WITH(executor.run_inference(input), Catch::Matchers::Contains("No model loaded"));
  }

  SECTION("Null pointer handling in GPU allocator")
  {
    OrtGpuMemoryAllocator allocator(0);

    // Deallocating nullptr should not crash
    REQUIRE_NOTHROW(allocator.deallocate(nullptr));

    // Copy operations with null should not crash (though may throw)
    REQUIRE_NOTHROW(allocator.copy_from_host(nullptr, nullptr, 0));
    REQUIRE_NOTHROW(allocator.copy_to_host(nullptr, nullptr, 0));
    REQUIRE_NOTHROW(allocator.copy_device_to_device(nullptr, nullptr, 0));
  }

  SECTION("Large allocation failure handling")
  {
    OrtGpuMemoryAllocator allocator(0);

    // Try to allocate an impossibly large amount of memory
    void * ptr = allocator.allocate(SIZE_MAX);
    REQUIRE(ptr == nullptr);  // Should return nullptr, not crash
  }
}

TEST_CASE("GPU capability verification", "[gpu][capability]")
{
  if (!is_cuda_available()) {
    return;
  }

  SECTION("Device capability check")
  {
    cudaDeviceProp prop;
    cudaError_t result = cudaGetDeviceProperties(&prop, 0);
    REQUIRE(result == cudaSuccess);

    // Verify minimum compute capability (3.5+)
    bool has_min_capability = prop.major > 3 || (prop.major == 3 && prop.minor >= 5);
    REQUIRE(has_min_capability);

    std::cout << "GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
  }

  SECTION("Memory info verification")
  {
    size_t free_mem, total_mem;
    cudaError_t result = cudaMemGetInfo(&free_mem, &total_mem);
    REQUIRE(result == cudaSuccess);

    REQUIRE(free_mem > 0);
    REQUIRE(total_mem > 0);
    REQUIRE(free_mem <= total_mem);

    std::cout << "GPU Memory: " << (free_mem / 1024 / 1024) << "MB free / " << (total_mem / 1024 / 1024) << "MB total"
              << std::endl;
  }
}

}  // namespace test
}  // namespace deep_ort_gpu_backend
