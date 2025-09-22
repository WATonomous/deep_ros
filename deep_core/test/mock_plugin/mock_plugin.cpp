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
    return std::malloc(bytes);
  }

  void deallocate(void * ptr) override
  {
    if (ptr) {
      std::free(ptr);
    }
  }

  void copy_from_host(void * dst, const void * src, size_t bytes) override
  {
    std::memcpy(dst, src, bytes);
  }

  void copy_to_host(void * dst, const void * src, size_t bytes) override
  {
    std::memcpy(dst, src, bytes);
  }

  void copy_device_to_device(void * dst, const void * src, size_t bytes) override
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
};

class MockInferenceExecutor : public BackendInferenceExecutor
{
public:
  bool load_model(const std::filesystem::path & model_path) override
  {
    model_loaded_ = true;
    current_model_path_ = model_path;
    return true;
  }

  Tensor run_inference(Tensor input) override
  {
    if (!model_loaded_) {
      throw std::runtime_error("No model loaded");
    }

    // For testing, just return a copy of the input
    return input;
  }

  void unload_model() override
  {
    model_loaded_ = false;
    current_model_path_.clear();
  }

  std::vector<std::string> supported_model_formats() const override
  {
    return {"mock", "dummy"};
  }

private:
  bool model_loaded_ = false;
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
