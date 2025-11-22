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
#include <string>
#include <vector>

namespace deep_ort_gpu_backend
{

// Static instance pointer for callbacks
OrtGpuCpuMemoryAllocator * OrtGpuCpuMemoryAllocator::instance_ = nullptr;

OrtGpuCpuMemoryAllocator::OrtGpuCpuMemoryAllocator()
: ort_memory_info_(nullptr)
{
  // Set the static instance pointer for callbacks
  instance_ = this;

  // Initialize OrtAllocator struct
  ort_allocator_.version = ORT_API_VERSION;
  ort_allocator_.Alloc = &OrtGpuCpuMemoryAllocator::ort_alloc;
  ort_allocator_.Free = &OrtGpuCpuMemoryAllocator::ort_free;
  ort_allocator_.Info = &OrtGpuCpuMemoryAllocator::ort_info;
  ort_allocator_.Reserve = &OrtGpuCpuMemoryAllocator::ort_reserve;

  // Create OrtMemoryInfo for CPU allocator
  OrtStatus * status = OrtGetApiBase()
                         ->GetApi(ORT_API_VERSION)
                         ->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort_memory_info_);
  if (status != nullptr) {
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
    throw std::runtime_error("Failed to create OrtMemoryInfo for GPU backend CPU allocator");
  }
}

OrtGpuCpuMemoryAllocator::~OrtGpuCpuMemoryAllocator()
{
  if (ort_memory_info_) {
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseMemoryInfo(ort_memory_info_);
  }
}

OrtAllocator * OrtGpuCpuMemoryAllocator::get_ort_allocator()
{
  return &ort_allocator_;
}

const OrtMemoryInfo * OrtGpuCpuMemoryAllocator::get_ort_memory_info() const
{
  return ort_memory_info_;
}

void * ORT_API_CALL OrtGpuCpuMemoryAllocator::ort_alloc(OrtAllocator * /*this_*/, size_t size)
{
  if (instance_) {
    return instance_->allocate(size);
  }
  return nullptr;
}

void ORT_API_CALL OrtGpuCpuMemoryAllocator::ort_free(OrtAllocator * /*this_*/, void * p)
{
  if (instance_) {
    instance_->deallocate(p);
  }
}

const OrtMemoryInfo * ORT_API_CALL OrtGpuCpuMemoryAllocator::ort_info(const OrtAllocator * /*this_*/)
{
  if (instance_) {
    return instance_->get_ort_memory_info();
  }
  return nullptr;
}

void * ORT_API_CALL OrtGpuCpuMemoryAllocator::ort_reserve(OrtAllocator * /*this_*/, size_t size)
{
  if (instance_) {
    return instance_->allocate(size);
  }
  return nullptr;
}

void * OrtGpuCpuMemoryAllocator::allocate(size_t bytes)
{
  if (bytes == 0) {
    return nullptr;
  }

  // Use 64-byte alignment for optimal SIMD performance
  void * ptr = std::aligned_alloc(64, bytes);
  if (!ptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void OrtGpuCpuMemoryAllocator::deallocate(void * ptr)
{
  if (ptr) {
    std::free(ptr);
  }
}

void OrtGpuCpuMemoryAllocator::copy_from_host_impl(void * dst, const void * src, size_t bytes)
{
  if (dst && src && bytes > 0) {
    std::memcpy(dst, src, bytes);
  }
}

void OrtGpuCpuMemoryAllocator::copy_from_host_permuted_impl(
  void * dst,
  const void * src,
  const std::vector<size_t> & src_shape,
  const std::vector<size_t> & permutation,
  size_t elem_size)
{
  // Full permutation implementation
  size_t total_elements = 1;
  for (size_t dim : src_shape) {
    total_elements *= dim;
  }

  std::vector<size_t> src_strides(src_shape.size(), 1);
  for (int i = static_cast<int>(src_shape.size()) - 2; i >= 0; --i) {
    src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
  }

  std::vector<size_t> dst_shape(src_shape.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    dst_shape[i] = src_shape[permutation[i]];
  }

  const auto * src_ptr = static_cast<const uint8_t *>(src);
  auto * dst_ptr = static_cast<uint8_t *>(dst);

  std::vector<size_t> dst_indices(dst_shape.size(), 0);

  for (size_t dst_linear = 0; dst_linear < total_elements; ++dst_linear) {
    size_t src_linear = 0;
    for (size_t i = 0; i < permutation.size(); ++i) {
      src_linear += dst_indices[i] * src_strides[permutation[i]];
    }

    std::memcpy(dst_ptr + dst_linear * elem_size, src_ptr + src_linear * elem_size, elem_size);

    for (int i = static_cast<int>(dst_shape.size()) - 1; i >= 0; --i) {
      if (++dst_indices[i] < dst_shape[i]) {
        break;
      }
      dst_indices[i] = 0;
    }
  }
}

void OrtGpuCpuMemoryAllocator::copy_to_host_impl(void * dst, const void * src, size_t bytes)
{
  if (dst && src && bytes > 0) {
    std::memcpy(dst, src, bytes);
  }
}

void OrtGpuCpuMemoryAllocator::copy_device_to_device_impl(void * dst, const void * src, size_t bytes)
{
  if (dst && src && bytes > 0) {
    std::memcpy(dst, src, bytes);
  }
}

bool OrtGpuCpuMemoryAllocator::is_device_memory() const
{
  return false;  // CPU memory is host memory
}

std::string OrtGpuCpuMemoryAllocator::device_name() const
{
  return "gpu_backend_cpu";  // Distinguish from regular CPU backend
}

std::shared_ptr<deep_ros::BackendMemoryAllocator> get_ort_gpu_cpu_allocator()
{
  static std::shared_ptr<deep_ros::BackendMemoryAllocator> allocator = std::make_shared<OrtGpuCpuMemoryAllocator>();
  return allocator;
}

}  // namespace deep_ort_gpu_backend
