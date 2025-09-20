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

#pragma once

#include <memory>
#include <string>

#include "deep_tensor/memory_allocator.hpp"

namespace onnxruntime
{

/**
 * @brief ONNX Runtime optimized CPU memory allocator
 *
 * Uses aligned memory allocation optimized for ONNX Runtime CPU operations.
 * Provides better performance than default CPU allocator for inference workloads.
 */
class OnnxCpuAllocator : public deep_ros::MemoryAllocator
{
public:
  OnnxCpuAllocator();
  ~OnnxCpuAllocator() override = default;

  void * allocate(size_t bytes) override;
  void deallocate(void * ptr) override;
  void copy_from_host(void * dst, const void * src, size_t bytes) override;
  void copy_to_host(void * dst, const void * src, size_t bytes) override;
  void copy_device_to_device(void * dst, const void * src, size_t bytes) override;
  bool is_device_memory() const override;
  std::string device_name() const override;

private:
  static constexpr size_t ALIGNMENT = 64;  // 64-byte alignment for AVX-512
};

/**
 * @brief Get shared instance of ONNX CPU allocator
 * @return Shared pointer to ONNX CPU allocator
 */
std::shared_ptr<deep_ros::MemoryAllocator> get_onnx_cpu_allocator();

}  // namespace onnxruntime
