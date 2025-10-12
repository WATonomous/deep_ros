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

#include "deep_core/plugin_interfaces/backend_memory_allocator.hpp"

#include <stdexcept>
#include <vector>

namespace deep_ros
{

void BackendMemoryAllocator::copy_from_host(void * dst, const void * src, size_t bytes)
{
  if (bytes > 0 && (dst == nullptr || src == nullptr)) {
    throw std::invalid_argument("Null pointer passed to copy_from_host");
  }
  copy_from_host_impl(dst, src, bytes);
}

void BackendMemoryAllocator::copy_from_host_permuted(
  void * dst,
  const void * src,
  const std::vector<size_t> & src_shape,
  const std::vector<size_t> & permutation,
  size_t elem_size)
{
  if (dst == nullptr || src == nullptr) {
    throw std::invalid_argument("Null pointer passed to copy_from_host_permuted");
  }
  if (src_shape.empty() || permutation.empty()) {
    throw std::invalid_argument("Empty shape or permutation passed to copy_from_host_permuted");
  }
  if (src_shape.size() != permutation.size()) {
    throw std::invalid_argument("Shape and permutation size mismatch in copy_from_host_permuted");
  }
  copy_from_host_permuted_impl(dst, src, src_shape, permutation, elem_size);
}

void BackendMemoryAllocator::copy_to_host(void * dst, const void * src, size_t bytes)
{
  if (bytes > 0 && (dst == nullptr || src == nullptr)) {
    throw std::invalid_argument("Null pointer passed to copy_to_host");
  }
  copy_to_host_impl(dst, src, bytes);
}

void BackendMemoryAllocator::copy_device_to_device(void * dst, const void * src, size_t bytes)
{
  if (bytes > 0 && (dst == nullptr || src == nullptr)) {
    throw std::invalid_argument("Null pointer passed to copy_device_to_device");
  }
  copy_device_to_device_impl(dst, src, bytes);
}

void * BackendMemoryAllocator::allocate(size_t bytes, size_t alignment)
{
  (void)bytes;  // Suppress unused parameter warning
  (void)alignment;  // Suppress unused parameter warning
  throw std::runtime_error("Aligned allocation not implemented for this backend");
}

}  // namespace deep_ros
