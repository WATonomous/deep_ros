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

// Copyright (c) 2025-present WATonomous. All rights reserved.
//
// Licensed under the Apacnse, Version 2.0 (the "License");
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

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <deep_core/plugin_interfaces/backend_memory_allocator.hpp>

namespace deep_ort_gpu_backend
{

/**
 * @brief Simple CPU memory allocator for output tensors
 */
class SimpleCpuAllocator : public deep_ros::BackendMemoryAllocator
{
public:
  void * allocate(size_t bytes) override
  {
    return std::malloc(bytes);
  }

  void deallocate(void * ptr) override
  {
    std::free(ptr);
  }

  bool is_device_memory() const override
  {
    return false;
  }

  std::string device_name() const override
  {
    return "cpu";
  }

protected:
  void copy_from_host_impl(void * dst, const void * src, size_t bytes) override
  {
    memcpy(dst, src, bytes);
  }

  void copy_from_host_permuted_impl(
    void * dst,
    const void * src,
    const std::vector<size_t> & src_shape,
    const std::vector<size_t> & permutation,
    size_t elem_size) override
  {
    // Simple implementation - just copy without permutation
    size_t total_elements = 1;
    for (size_t dim : src_shape) {
      total_elements *= dim;
    }
    memcpy(dst, src, total_elements * elem_size);
  }

  void copy_to_host_impl(void * dst, const void * src, size_t bytes) override
  {
    memcpy(dst, src, bytes);
  }

  void copy_device_to_device_impl(void * dst, const void * src, size_t bytes) override
  {
    memcpy(dst, src, bytes);
  }
};

std::shared_ptr<deep_ros::BackendMemoryAllocator> get_simple_cpu_allocator();

}  // namespace deep_ort_gpu_backend
