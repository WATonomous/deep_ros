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

#include "deep_ort_backend_plugin/ort_cpu_memory_allocator.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace deep_ort_backend
{

OrtCpuMemoryAllocator::OrtCpuMemoryAllocator()
{
  // TODO(wato): Initialize with ONNX Runtime CPU allocator if available
}

void * OrtCpuMemoryAllocator::allocate(size_t bytes)
{
  if (bytes == 0) {
    return nullptr;
  }

  // Use 64-byte alignment for optimal SIMD performance (AVX-512)
  void * ptr = std::aligned_alloc(64, bytes);
  if (!ptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void OrtCpuMemoryAllocator::deallocate(void * ptr)
{
  if (ptr) {
    std::free(ptr);
  }
}

void OrtCpuMemoryAllocator::copy_from_host(void * dst, const void * src, size_t bytes)
{
  if (dst && src && bytes > 0) {
    std::memcpy(dst, src, bytes);
  }
}

void OrtCpuMemoryAllocator::copy_to_host(void * dst, const void * src, size_t bytes)
{
  if (dst && src && bytes > 0) {
    std::memcpy(dst, src, bytes);
  }
}

void OrtCpuMemoryAllocator::copy_device_to_device(void * dst, const void * src, size_t bytes)
{
  if (dst && src && bytes > 0) {
    std::memcpy(dst, src, bytes);
  }
}

bool OrtCpuMemoryAllocator::is_device_memory() const
{
  return false;  // CPU memory is host memory
}

std::string OrtCpuMemoryAllocator::device_name() const
{
  return "ort_cpu";
}

std::shared_ptr<deep_ros::BackendMemoryAllocator> get_ort_cpu_allocator()
{
  static std::shared_ptr<deep_ros::BackendMemoryAllocator> allocator = std::make_shared<OrtCpuMemoryAllocator>();
  return allocator;
}

}  // namespace deep_ort_backend
