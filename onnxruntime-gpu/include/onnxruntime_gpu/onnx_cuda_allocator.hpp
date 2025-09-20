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

namespace onnxruntime_gpu
{

/**
 * @brief ONNX Runtime CUDA memory allocator
 *
 * Uses CUDA Runtime API for GPU memory allocation optimized for ONNX Runtime.
 * Provides efficient GPU memory management for inference workloads.
 */
class OnnxCudaAllocator : public deep_ros::MemoryAllocator
{
public:
  /**
   * @brief Construct CUDA allocator for specific device
   * @param device_id GPU device ID (default: 0)
   */
  explicit OnnxCudaAllocator(int device_id = 0);
  ~OnnxCudaAllocator() override;

  void * allocate(size_t bytes) override;
  void deallocate(void * ptr) override;
  void copy_from_host(void * dst, const void * src, size_t bytes) override;
  void copy_to_host(void * dst, const void * src, size_t bytes) override;
  void copy_device_to_device(void * dst, const void * src, size_t bytes) override;
  bool is_device_memory() const override;
  std::string device_name() const override;

  /**
   * @brief Get the CUDA device ID
   * @return Device ID
   */
  int device_id() const
  {
    return device_id_;
  }

  /**
   * @brief Check if CUDA is available on this system
   * @return true if CUDA runtime is available and devices detected
   */
  static bool is_cuda_available();

private:
  int device_id_;

#ifdef ONNX_GPU_CUDA_ENABLED
  void check_cuda_error(int error, const std::string & operation);
#endif
};

/**
 * @brief Get shared instance of ONNX CUDA allocator
 * @param device_id GPU device ID (default: 0)
 * @return Shared pointer to ONNX CUDA allocator, or nullptr if CUDA not available
 */
std::shared_ptr<deep_ros::MemoryAllocator> get_onnx_cuda_allocator(int device_id = 0);

}  // namespace onnxruntime_gpu
