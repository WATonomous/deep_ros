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

#include "deep_tensor/memory_allocator.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace deep_ros
{

void * CpuAllocator::allocate(size_t bytes)
{
  return std::malloc(bytes);
}

void CpuAllocator::deallocate(void * ptr)
{
  std::free(ptr);
}

void CpuAllocator::copy_from_host(void * dst, const void * src, size_t bytes)
{
  std::memcpy(dst, src, bytes);
}

void CpuAllocator::copy_to_host(void * dst, const void * src, size_t bytes)
{
  std::memcpy(dst, src, bytes);
}

void CpuAllocator::copy_device_to_device(void * dst, const void * src, size_t bytes)
{
  std::memcpy(dst, src, bytes);
}

bool CpuAllocator::is_device_memory() const
{
  return false;
}

std::string CpuAllocator::device_name() const
{
  return "cpu";
}

std::shared_ptr<MemoryAllocator> get_cpu_allocator()
{
  static auto instance = std::make_shared<CpuAllocator>();
  return instance;
}

}  // namespace deep_ros
