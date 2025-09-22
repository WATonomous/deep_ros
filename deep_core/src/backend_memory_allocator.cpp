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

namespace deep_ros
{

void BackendMemoryAllocator::copy_from_host(void * dst, const void * src, size_t bytes)
{
  if (bytes > 0 && (dst == nullptr || src == nullptr)) {
    throw std::invalid_argument("Null pointer passed to copy_from_host");
  }
  copy_from_host_impl(dst, src, bytes);
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

}  // namespace deep_ros
