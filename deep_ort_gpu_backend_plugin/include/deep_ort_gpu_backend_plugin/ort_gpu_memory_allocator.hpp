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
 * @brief CPU memory allocator for GPU backend output tensors (similar to CPU backend)
 */
class OrtGpuCpuMemoryAllocator : public deep_ros::BackendMemoryAllocator
{
public:
  /**
   * @brief Constructor - initializes ORT allocator integration
   */
  OrtGpuCpuMemoryAllocator();

  /**
   * @brief Destructor - cleans up ORT resources
   */
  ~OrtGpuCpuMemoryAllocator() override;

  /**
   * @brief Get the ORT allocator for integration
   * @return Pointer to OrtAllocator
   */
  OrtAllocator * get_ort_allocator();

  /**
   * @brief Get the ORT memory info
   * @return Pointer to OrtMemoryInfo
   */
  const OrtMemoryInfo * get_ort_memory_info() const;

  // BackendMemoryAllocator interface implementation
  void * allocate(size_t bytes) override;
  void deallocate(void * ptr) override;
  bool is_device_memory() const override;
  std::string device_name() const override;

protected:
  void copy_from_host_impl(void * dst, const void * src, size_t bytes) override;
  void copy_from_host_permuted_impl(
    void * dst,
    const void * src,
    const std::vector<size_t> & src_shape,
    const std::vector<size_t> & permutation,
    size_t elem_size) override;
  void copy_to_host_impl(void * dst, const void * src, size_t bytes) override;
  void copy_device_to_device_impl(void * dst, const void * src, size_t bytes) override;

private:
  // ORT allocator integration (static instance for callbacks)
  static OrtGpuCpuMemoryAllocator * instance_;
  OrtAllocator ort_allocator_;
  OrtMemoryInfo * ort_memory_info_;

  // Static callback functions for ORT integration
  static void * ORT_API_CALL ort_alloc(OrtAllocator * this_, size_t size);
  static void ORT_API_CALL ort_free(OrtAllocator * this_, void * p);
  static const OrtMemoryInfo * ORT_API_CALL ort_info(const OrtAllocator * this_);
  static void * ORT_API_CALL ort_reserve(OrtAllocator * this_, size_t size);
};

std::shared_ptr<deep_ros::BackendMemoryAllocator> get_ort_gpu_cpu_allocator();

}  // namespace deep_ort_gpu_backend
