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

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pluginlib/class_list_macros.hpp>

#include "deep_core/plugin_interfaces/backend_inference_executor.hpp"
#include "deep_core/plugin_interfaces/backend_memory_allocator.hpp"
#include "deep_core/plugin_interfaces/deep_backend_plugin.hpp"

namespace deep_ros
{
namespace mock
{

class MockMemoryAllocator : public BackendMemoryAllocator
{
public:
  void * allocate(size_t bytes) override
  {
    if (bytes == 0) return nullptr;
    void * ptr = std::malloc(bytes);
    if (ptr) {
      allocated_bytes_ += bytes;
    }
    return ptr;
  }

  void deallocate(void * ptr) override
  {
    if (ptr) {
      std::free(ptr);
    }
  }

  void copy_from_host_impl(void * dst, const void * src, size_t bytes) override
  {
    std::memcpy(dst, src, bytes);
  }

  void copy_from_host_permuted_impl(
    void * dst,
    const void * src,
    const std::vector<size_t> & src_shape,
    const std::vector<size_t> & permutation,
    size_t elem_size) override
  {
    // Calculate total elements
    size_t total_elements = 1;
    for (size_t dim : src_shape) {
      total_elements *= dim;
    }

    // Compute strides for source dimensions
    std::vector<size_t> src_strides(src_shape.size(), 1);
    for (int i = static_cast<int>(src_shape.size()) - 2; i >= 0; --i) {
      src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }

    // Compute destination shape
    std::vector<size_t> dst_shape(src_shape.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
      dst_shape[i] = src_shape[permutation[i]];
    }

    const auto * src_ptr = static_cast<const uint8_t *>(src);
    auto * dst_ptr = static_cast<uint8_t *>(dst);

    // Perform permutation during copy
    std::vector<size_t> dst_indices(dst_shape.size(), 0);

    for (size_t dst_linear = 0; dst_linear < total_elements; ++dst_linear) {
      // Map destination indices to source linear index
      size_t src_linear = 0;
      for (size_t i = 0; i < permutation.size(); ++i) {
        src_linear += dst_indices[i] * src_strides[permutation[i]];
      }

      // Copy element
      std::memcpy(dst_ptr + dst_linear * elem_size, src_ptr + src_linear * elem_size, elem_size);

      // Increment destination indices
      for (int i = static_cast<int>(dst_shape.size()) - 1; i >= 0; --i) {
        if (++dst_indices[i] < dst_shape[i]) {
          break;
        }
        dst_indices[i] = 0;
      }
    }
  }

  void copy_to_host_impl(void * dst, const void * src, size_t bytes) override
  {
    std::memcpy(dst, src, bytes);
  }

  void copy_device_to_device_impl(void * dst, const void * src, size_t bytes) override
  {
    std::memcpy(dst, src, bytes);
  }

  bool is_device_memory() const override
  {
    return false;
  }

  std::string device_name() const override
  {
    return "mock_cpu";
  }

  size_t allocated_bytes() const
  {
    return allocated_bytes_;
  }

  void reset_allocation_counter()
  {
    allocated_bytes_ = 0;
  }

private:
  size_t allocated_bytes_{0};
};

class MockInferenceExecutor : public BackendInferenceExecutor
{
public:
  bool load_model_impl(const std::filesystem::path & model_path) override
  {
    current_model_path_ = model_path;
    return true;
  }

  Tensor run_inference_impl(Tensor & input) override
  {
    // For testing, create a simple output tensor with same shape/dtype using input's allocator
    return Tensor(input.shape(), input.dtype(), input.allocator());
  }

  void unload_model_impl() override
  {
    current_model_path_.clear();
  }

  std::vector<std::string> supported_model_formats() const override
  {
    return {"mock", "dummy"};
  }

private:
  std::filesystem::path current_model_path_;
};

class MockBackendPlugin : public DeepBackendPlugin
{
public:
  MockBackendPlugin()
  {
    allocator_ = std::make_shared<MockMemoryAllocator>();
    executor_ = std::make_shared<MockInferenceExecutor>();
  }

  std::string backend_name() const override
  {
    return "mock_backend";
  }

  std::shared_ptr<BackendMemoryAllocator> get_allocator() const override
  {
    return allocator_;
  }

  std::shared_ptr<BackendInferenceExecutor> get_inference_executor() const override
  {
    return executor_;
  }

private:
  std::shared_ptr<MockMemoryAllocator> allocator_;
  std::shared_ptr<MockInferenceExecutor> executor_;
};

}  // namespace mock
}  // namespace deep_ros

PLUGINLIB_EXPORT_CLASS(deep_ros::mock::MockBackendPlugin, deep_ros::DeepBackendPlugin)
