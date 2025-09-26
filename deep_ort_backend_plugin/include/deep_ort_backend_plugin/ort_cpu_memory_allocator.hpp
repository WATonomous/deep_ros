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

#include <onnxruntime_c_api.h>

#include <memory>
#include <string>

#include <deep_core/plugin_interfaces/backend_memory_allocator.hpp>

namespace deep_ort_backend
{

/**
 * @brief ONNX Runtime optimized CPU memory allocator
 *
 * Provides CPU memory allocation optimized for ONNX Runtime operations
 * with proper alignment for SIMD operations. Implements both deep_ros
 * BackendMemoryAllocator interface and OrtAllocator interface directly.
 */
class OrtCpuMemoryAllocator : public deep_ros::BackendMemoryAllocator
{
public:
  /**
   * @brief Constructor
   */
  OrtCpuMemoryAllocator();

  /**
   * @brief Destructor
   */
  ~OrtCpuMemoryAllocator() override;

  /**
   * @brief Get the OrtAllocator interface for use with ONNX Runtime
   * @return Pointer to OrtAllocator struct
   */
  OrtAllocator * get_ort_allocator();

  /**
   * @brief Get the OrtMemoryInfo for this allocator
   * @return Pointer to OrtMemoryInfo
   */
  const OrtMemoryInfo * get_ort_memory_info() const;

  /**
   * @brief Allocate aligned memory for CPU operations
   * @param bytes Number of bytes to allocate
   * @return Pointer to 64-byte aligned memory, or nullptr on failure
   */
  void * allocate(size_t bytes) override;

  /**
   * @brief Deallocate memory
   * @param ptr Pointer to memory allocated by this allocator
   */
  void deallocate(void * ptr) override;

  /**
   * @brief Check if this is device memory
   * @return false (CPU memory is host memory)
   */
  bool is_device_memory() const override;

  /**
   * @brief Get device name
   * @return "cpu"
   */
  std::string device_name() const override;

protected:
  /**
   * @brief Copy from host memory (same as device for CPU)
   * @param dst Destination pointer
   * @param src Source pointer
   * @param bytes Number of bytes to copy
   */
  void copy_from_host_impl(void * dst, const void * src, size_t bytes) override;

  /**
   * @brief Copy to host memory (same as device for CPU)
   * @param dst Destination pointer
   * @param src Source pointer
   * @param bytes Number of bytes to copy
   */
  void copy_to_host_impl(void * dst, const void * src, size_t bytes) override;

  /**
   * @brief Copy between CPU memory locations
   * @param dst Destination pointer
   * @param src Source pointer
   * @param bytes Number of bytes to copy
   */
  void copy_device_to_device_impl(void * dst, const void * src, size_t bytes) override;

private:
  OrtAllocator ort_allocator_;
  OrtMemoryInfo * ort_memory_info_;

  // Store a pointer to self in a way that callbacks can access it
  static OrtCpuMemoryAllocator * instance_;

  // Static callback functions for OrtAllocator interface
  static void * ORT_API_CALL ort_alloc(OrtAllocator * this_, size_t size);
  static void ORT_API_CALL ort_free(OrtAllocator * this_, void * p);
  static const OrtMemoryInfo * ORT_API_CALL ort_info(const OrtAllocator * this_);
  static void * ORT_API_CALL ort_reserve(OrtAllocator * this_, size_t size);
};

/**
 * @brief Get a shared instance of the ORT CPU allocator
 * @return Shared pointer to ORT CPU allocator
 */
std::shared_ptr<deep_ros::BackendMemoryAllocator> get_ort_cpu_allocator();

}  // namespace deep_ort_backend
