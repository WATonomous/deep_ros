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

namespace deep_ros
{

/**
 * @brief Abstract interface for memory allocation strategies
 *
 * This interface allows different backends (CPU, CUDA, etc.) to provide
 * their optimal memory allocation and copying strategies.
 */
class MemoryAllocator
{
public:
  virtual ~MemoryAllocator() = default;

  /**
   * @brief Allocate memory
   * @param bytes Number of bytes to allocate
   * @return Pointer to allocated memory, nullptr on failure
   */
  virtual void * allocate(size_t bytes) = 0;

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
   */
  virtual void copy_from_host(void * dst, const void * src, size_t bytes) = 0;

  /**
   * @brief Copy data from allocated memory to host (CPU) memory
   * @param dst Destination pointer (host memory)
   * @param src Source pointer (allocated by this allocator)
   * @param bytes Number of bytes to copy
   */
  virtual void copy_to_host(void * dst, const void * src, size_t bytes) = 0;

  /**
   * @brief Copy data between two allocations from this allocator
   * @param dst Destination pointer (allocated by this allocator)
   * @param src Source pointer (allocated by this allocator)
   * @param bytes Number of bytes to copy
   */
  virtual void copy_device_to_device(void * dst, const void * src, size_t bytes) = 0;

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
};

/**
 * @brief Default CPU memory allocator
 *
 * Uses standard malloc/free for allocation and memcpy for copying.
 * Always available as fallback.
 */
class CpuAllocator : public MemoryAllocator
{
public:
  void * allocate(size_t bytes) override;
  void deallocate(void * ptr) override;
  void copy_from_host(void * dst, const void * src, size_t bytes) override;
  void copy_to_host(void * dst, const void * src, size_t bytes) override;
  void copy_device_to_device(void * dst, const void * src, size_t bytes) override;
  bool is_device_memory() const override;
  std::string device_name() const override;
};

/**
 * @brief Get the default CPU allocator instance
 * @return Shared pointer to CPU allocator
 */
std::shared_ptr<MemoryAllocator> get_cpu_allocator();

}  // namespace deep_ros
