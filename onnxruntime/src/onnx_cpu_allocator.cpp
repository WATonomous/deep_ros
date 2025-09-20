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

#include "onnxruntime/onnx_cpu_allocator.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace onnxruntime
{

OnnxCpuAllocator::OnnxCpuAllocator()
{
  // Initialize ONNX Runtime CPU allocator here when integrating with actual ONNX Runtime
}

void * OnnxCpuAllocator::allocate(size_t bytes)
{
  // Use aligned allocation optimized for CPU SIMD operations
  void * ptr = std::aligned_alloc(ALIGNMENT, (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1));
  if (!ptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void OnnxCpuAllocator::deallocate(void * ptr)
{
  if (ptr) {
    std::free(ptr);
  }
}

void OnnxCpuAllocator::copy_from_host(void * dst, const void * src, size_t bytes)
{
  // Optimized memory copy - could use SIMD or other optimizations
  std::memcpy(dst, src, bytes);
}

void OnnxCpuAllocator::copy_to_host(void * dst, const void * src, size_t bytes)
{
  // Same as copy_from_host for CPU allocator
  std::memcpy(dst, src, bytes);
}

void OnnxCpuAllocator::copy_device_to_device(void * dst, const void * src, size_t bytes)
{
  // CPU-to-CPU copy
  std::memcpy(dst, src, bytes);
}

bool OnnxCpuAllocator::is_device_memory() const
{
  return false;  // CPU memory
}

std::string OnnxCpuAllocator::device_name() const
{
  return "onnx_cpu";
}

std::shared_ptr<deep_ros::MemoryAllocator> get_onnx_cpu_allocator()
{
  static auto instance = std::make_shared<OnnxCpuAllocator>();
  return instance;
}

}  // namespace onnxruntime
