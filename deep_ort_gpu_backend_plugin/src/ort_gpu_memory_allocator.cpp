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

#include "deep_ort_gpu_backend_plugin/ort_gpu_memory_allocator.hpp"

#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace deep_ort_gpu_backend
{

// thread_local OrtGpuMemoryAllocator * OrtGpuMemoryAllocator::instance_ = nullptr;

// OrtGpuMemoryAllocator::OrtGpuMemoryAllocator(int device_id)
// : device_id_(device_id), ort_memory_info_(nullptr)
// {
//   if (!initialize_cuda()) {
//     throw std::runtime_error("Failed to initialize CUDA runtime");
//   }

//   set_device();

//   // Initialize ORT memory info for CUDA
//   const OrtApi * api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
//   OrtStatus * status = api->CreateMemoryInfo(
//     "Cuda", OrtAllocatorType::OrtArenaAllocator, device_id_,
//     OrtMemType::OrtMemTypeDefault, &ort_memory_info_);

//   if (status != nullptr) {
//     api->ReleaseStatus(status);
//     throw std::runtime_error("Failed to create ORT memory info for CUDA");
//   }

//   // Initialize ORT allocator callbacks
//   ort_allocator_.version = ORT_API_VERSION;
//   ort_allocator_.Alloc = &OrtGpuMemoryAllocator::ort_alloc;
//   ort_allocator_.Free = &OrtGpuMemoryAllocator::ort_free;
//   ort_allocator_.Info = &OrtGpuMemoryAllocator::ort_info;
//   ort_allocator_.Reserve = &OrtGpuMemoryAllocator::ort_reserve;

//   instance_ = this;
// }

// OrtGpuMemoryAllocator::~OrtGpuMemoryAllocator()
// {
//   if (ort_memory_info_) {
//     const OrtApi * api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
//     api->ReleaseMemoryInfo(ort_memory_info_);
//   }
//   instance_ = nullptr;
// }

// OrtAllocator * OrtGpuMemoryAllocator::get_ort_allocator()
// {
//   return &ort_allocator_;
// }

// const OrtMemoryInfo * OrtGpuMemoryAllocator::get_ort_memory_info() const
// {
//   return ort_memory_info_;
// }

// void * OrtGpuMemoryAllocator::allocate(size_t bytes)
// {
//   if (bytes == 0) {
//     return nullptr;
//   }

//   set_device();

//   void * ptr = nullptr;
//   cudaError_t result = cudaMalloc(&ptr, bytes);

//   if (result != cudaSuccess) {
//     return nullptr;
//   }

//   return ptr;
// }

// void OrtGpuMemoryAllocator::deallocate(void * ptr)
// {
//   if (ptr == nullptr) {
//     return;
//   }

//   set_device();
//   cudaFree(ptr);
// }

// bool OrtGpuMemoryAllocator::is_device_memory() const
// {
//   return true;
// }

// std::string OrtGpuMemoryAllocator::device_name() const
// {
//   return "cuda:" + std::to_string(device_id_);
// }

// int OrtGpuMemoryAllocator::get_device_id() const
// {
//   return device_id_;
// }

// void OrtGpuMemoryAllocator::copy_from_host_impl(void * dst, const void * src, size_t bytes)
// {
//   if (dst == nullptr || src == nullptr || bytes == 0) {
//     return;
//   }

//   set_device();

//   cudaError_t result = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
//   if (result != cudaSuccess) {
//     throw std::runtime_error("CUDA memcpy host to device failed: " + std::string(cudaGetErrorString(result)));
//   }
// }

// void OrtGpuMemoryAllocator::copy_from_host_permuted_impl(
//   void * dst,
//   const void * src,
//   const std::vector<size_t> & src_shape,
//   const std::vector<size_t> & permutation,
//   size_t elem_size)
// {
//   if (dst == nullptr || src == nullptr || src_shape.empty() || permutation.empty()) {
//     return;
//   }

//   set_device();

//   // For permuted copies, we need to do this on CPU first, then copy to GPU
//   // Calculate total elements
//   size_t total_elements = 1;
//   for (size_t dim : src_shape) {
//     total_elements *= dim;
//   }

//   // Allocate temporary host buffer for permuted data
//   std::vector<uint8_t> temp_buffer(total_elements * elem_size);

//   // Perform permutation on CPU
//   // ...permutation logic would go here...
//   // For now, do a simple copy as placeholder
//   std::memcpy(temp_buffer.data(), src, total_elements * elem_size);

//   // Copy permuted data to GPU
//   copy_from_host_impl(dst, temp_buffer.data(), total_elements * elem_size);
// }

// void OrtGpuMemoryAllocator::copy_to_host_impl(void * dst, const void * src, size_t bytes)
// {
//   if (dst == nullptr || src == nullptr || bytes == 0) {
//     return;
//   }

//   set_device();

//   cudaError_t result = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
//   if (result != cudaSuccess) {
//     throw std::runtime_error("CUDA memcpy device to host failed: " + std::string(cudaGetErrorString(result)));
//   }
// }

// void OrtGpuMemoryAllocator::copy_device_to_device_impl(void * dst, const void * src, size_t bytes)
// {
//   if (dst == nullptr || src == nullptr || bytes == 0) {
//     return;
//   }

//   set_device();

//   cudaError_t result = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
//   if (result != cudaSuccess) {
//     throw std::runtime_error("CUDA memcpy device to device failed: " + std::string(cudaGetErrorString(result)));
//   }
// }

// void OrtGpuMemoryAllocator::set_device() const
// {
//   cudaError_t result = cudaSetDevice(device_id_);
//   if (result != cudaSuccess) {
//     throw std::runtime_error("Failed to set CUDA device: " + std::string(cudaGetErrorString(result)));
//   }
// }

// bool OrtGpuMemoryAllocator::initialize_cuda()
// {
//   // Check if CUDA is available
//   int device_count = 0;
//   cudaError_t result = cudaGetDeviceCount(&device_count);
//   if (result != cudaSuccess || device_count == 0) {
//     return false;
//   }

//   // Check if requested device exists
//   if (device_id_ >= device_count) {
//     return false;
//   }

//   // Initialize CUDA context
//   result = cudaSetDevice(device_id_);
//   if (result != cudaSuccess) {
//     return false;
//   }

//   // Force context initialization
//   result = cudaFree(0);
//   return result == cudaSuccess;
// }

// // Static callback functions
// void * ORT_API_CALL OrtGpuMemoryAllocator::ort_alloc(OrtAllocator * /*this_*/, size_t size)
// {
//   if (instance_ == nullptr) {
//     return nullptr;
//   }
//   return instance_->allocate(size);
// }

// void ORT_API_CALL OrtGpuMemoryAllocator::ort_free(OrtAllocator * /*this_*/, void * p)
// {
//   if (instance_ != nullptr) {
//     instance_->deallocate(p);
//   }
// }

// const OrtMemoryInfo * ORT_API_CALL OrtGpuMemoryAllocator::ort_info(const OrtAllocator * /*this_*/)
// {
//   if (instance_ == nullptr) {
//     return nullptr;
//   }
//   return instance_->get_ort_memory_info();
// }

// void * ORT_API_CALL OrtGpuMemoryAllocator::ort_reserve(OrtAllocator * this_, size_t size)
// {
//   return ort_alloc(this_, size);
// }

// std::shared_ptr<deep_ros::BackendMemoryAllocator> get_ort_gpu_allocator(int device_id)
// {
//   static std::shared_ptr<OrtGpuMemoryAllocator> instance;
//   if (!instance) {
//     instance = std::make_shared<OrtGpuMemoryAllocator>(device_id);
//   }
//   return instance;
// }

std::shared_ptr<deep_ros::BackendMemoryAllocator> get_simple_cpu_allocator()
{
  static std::shared_ptr<SimpleCpuAllocator> instance = std::make_shared<SimpleCpuAllocator>();
  return instance;
}

}  // namespace deep_ort_gpu_backend
