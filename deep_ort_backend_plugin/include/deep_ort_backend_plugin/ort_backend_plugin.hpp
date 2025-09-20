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

#include <deep_core/plugin_interfaces/backend_inference_executor.hpp>
#include <deep_core/plugin_interfaces/backend_memory_allocator.hpp>
#include <deep_core/plugin_interfaces/deep_backend_plugin.hpp>

namespace deep_ort_backend
{

/**
 * @brief ONNX Runtime backend plugin
 *
 * Combines ORT CPU memory allocator and inference executor into a single
 * backend plugin for use with pluginlib.
 */
class OrtBackendPlugin : public deep_ros::DeepBackendPlugin
{
public:
  OrtBackendPlugin();
  ~OrtBackendPlugin() override = default;

  std::string backend_name() const override;
  std::shared_ptr<deep_ros::BackendMemoryAllocator> get_allocator() const override;
  std::shared_ptr<deep_ros::BackendInferenceExecutor> get_inference_executor() const override;

private:
  std::shared_ptr<deep_ros::BackendMemoryAllocator> allocator_;
  std::shared_ptr<deep_ros::BackendInferenceExecutor> executor_;
};

}  // namespace deep_ort_backend
