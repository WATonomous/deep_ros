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

#include <deep_core/plugin_interfaces/deep_backend_plugin.hpp>
#include <deep_core/types/tensor.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "deep_object_detection/detection_types.hpp"

namespace deep_ros
{
class BackendInferenceExecutor;
class BackendMemoryAllocator;
}  // namespace deep_ros

namespace deep_object_detection
{

class BackendManager
{
public:
  BackendManager(rclcpp_lifecycle::LifecycleNode & node, const DetectionParams & params);
  ~BackendManager();

  void initialize();
  deep_ros::Tensor infer(const PackedInput & input);
  std::vector<deep_ros::Tensor> inferAllOutputs(const PackedInput & input);

  const std::string & activeProvider() const
  {
    return active_provider_;
  }

  std::shared_ptr<deep_ros::BackendMemoryAllocator> allocator() const
  {
    return allocator_;
  }

  bool hasExecutor() const
  {
    return static_cast<bool>(executor_);
  }

  std::vector<size_t> getOutputShape(const std::vector<size_t> & input_shape) const;

private:
  void initializeBackend();
  void warmupTensorShapeCache();
  Provider parseProvider(const std::string & provider_str) const;
  bool isCudaRuntimeAvailable() const;
  std::string providerToString(Provider provider) const;
  void declareActiveProviderParameter(const std::string & value);
  deep_ros::Tensor buildInputTensor(const PackedInput & packed) const;

  rclcpp_lifecycle::LifecycleNode & node_;
  const DetectionParams & params_;
  Provider provider_;
  std::string active_provider_{"unknown"};
  std::shared_ptr<deep_ros::BackendInferenceExecutor> executor_;
  std::shared_ptr<deep_ros::BackendMemoryAllocator> allocator_;
  std::unique_ptr<pluginlib::ClassLoader<deep_ros::DeepBackendPlugin>> plugin_loader_;
  pluginlib::UniquePtr<deep_ros::DeepBackendPlugin> plugin_holder_;

  std::string providerToPluginName(Provider provider) const;
};

}  // namespace deep_object_detection
