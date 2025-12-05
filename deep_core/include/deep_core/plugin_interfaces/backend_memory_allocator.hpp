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
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace deep_ros
{

/**
 * @brief Backend plugin interface for memory allocation strategies
 *
 * This plugin interface allows different backends (CPU, CUDA, etc.) to provide
 * their optimal memory allocation and copying strategies.
 */
class BackendMemoryAllocator
{
public:
  virtual ~BackendMemoryAllocator() = default;

  /**
   * @brief Allocate memory
   * @param bytes Number of bytes to allocate
   * @return Pointer to allocated memory, nullptr on failure
   */
  virtual void * allocate(size_t bytes) = 0;

  /**
   * @brief Allocate aligned memory
   * @param bytes Number of bytes to allocate
   * @param alignment Alignment requirement in bytes (must be power of 2)
   * @return Pointer to allocated memory, nullptr on failure
   */
  virtual void * allocate(size_t bytes, size_t alignment);

  /**
   * @brief Deallocate memory
   * @param ptr Pointer to memory allocated by this allocator
   */
  virtual void deallocate(void * ptr) = 0;

  /**
   * @brief Copy data from host (CPU) memory to allocated memory
   * @param dst Destination pointer (allocated by this allocator)
   * @param src Source pointer (host memory)
   * @param bytes Number of bytes to copy
   * @throws std::invalid_argument if dst or src is nullptr and bytes > 0
   */
  void copy_from_host(void * dst, const void * src, size_t bytes);

  /**
   * @brief Copy data from host memory to allocated memory with permutation
   *
   * Copies and transposes data in a single operation.
   *
   * @param dst Destination pointer (allocated by this allocator)
   * @param src Source pointer (host memory)
   * @param src_shape Shape of the source data
   * @param permutation Dimension permutation to apply during copy
   * @param elem_size Size of each element in bytes
   * @throws std::invalid_argument if parameters are invalid
   */
  void copy_from_host_permuted(
    void * dst,
    const void * src,
    const std::vector<size_t> & src_shape,
    const std::vector<size_t> & permutation,
    size_t elem_size);

  /**
   * @brief Copy data from allocated memory to host (CPU) memory
   * @param dst Destination pointer (host memory)
   * @param src Source pointer (allocated by this allocator)
   * @param bytes Number of bytes to copy
   * @throws std::invalid_argument if dst or src is nullptr and bytes > 0
   */
  void copy_to_host(void * dst, const void * src, size_t bytes);

  /**
   * @brief Copy data between two allocations from this allocator
   * @param dst Destination pointer (allocated by this allocator)
   * @param src Source pointer (allocated by this allocator)
   * @param bytes Number of bytes to copy
   * @throws std::invalid_argument if dst or src is nullptr and bytes > 0
   */
  void copy_device_to_device(void * dst, const void * src, size_t bytes);

protected:
  /**
   * @brief Implementation of copy_from_host (to be overridden by backends)
   */
  virtual void copy_from_host_impl(void * dst, const void * src, size_t bytes) = 0;

  /**
   * @brief Implementation of copy_from_host_permuted (to be overridden by backends)
   */
  virtual void copy_from_host_permuted_impl(
    void * dst,
    const void * src,
    const std::vector<size_t> & src_shape,
    const std::vector<size_t> & permutation,
    size_t elem_size) = 0;

  /**
   * @brief Implementation of copy_to_host (to be overridden by backends)
   */
  virtual void copy_to_host_impl(void * dst, const void * src, size_t bytes) = 0;

  /**
   * @brief Implementation of copy_device_to_device (to be overridden by backends)
   */
  virtual void copy_device_to_device_impl(void * dst, const void * src, size_t bytes) = 0;

public:
  /**
   * @brief Check if this allocator manages device (non-host) memory
   * @return true if memory is on device (GPU, etc.), false if host memory
   */
  virtual bool is_device_memory() const = 0;

  /**
   * @brief Get human-readable device name
   * @return Device name (e.g., "cpu", "cuda", "opencl")
   */
  virtual std::string device_name() const = 0;

  /**
   * @brief Get total bytes allocated by this allocator
   * @return Number of bytes currently allocated (for testing/debugging)
   */
  virtual size_t allocated_bytes() const
  {
    return 0;
  }
};

}  // namespace deep_ros
