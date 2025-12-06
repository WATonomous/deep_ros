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

#include <deep_core/types/tensor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "deep_yolo_inference/yolo_types.hpp"

namespace deep_ros
{
class BackendInferenceExecutor;
class BackendMemoryAllocator;
class DeepBackendPlugin;
}  // namespace deep_ros

namespace deep_yolo_inference
{

class BackendManager
{
public:
  BackendManager(rclcpp::Node & node, const YoloParams & params);

  void buildProviderOrder();
  bool initialize(size_t start_index = 0);
  deep_ros::Tensor infer(const PackedInput & input);
  bool fallbackToNextProvider();

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

private:
  bool initializeBackend(size_t start_index);
  void warmupTensorShapeCache(Provider provider);
  bool isCudaRuntimeAvailable() const;
  std::string providerToString(Provider provider) const;
  rclcpp_lifecycle::LifecycleNode::SharedPtr createBackendConfigNode(
    const std::string & suffix, std::vector<rclcpp::Parameter> overrides = {}) const;
  void declareActiveProviderParameter(const std::string & value);
  deep_ros::Tensor buildInputTensor(const PackedInput & packed) const;

  rclcpp::Node & node_;
  const YoloParams & params_;
  std::vector<Provider> provider_order_;
  size_t active_provider_index_{0};
  std::string active_provider_{"unknown"};
  std::shared_ptr<deep_ros::BackendInferenceExecutor> executor_;
  std::shared_ptr<deep_ros::BackendMemoryAllocator> allocator_;
  std::shared_ptr<deep_ros::DeepBackendPlugin> plugin_holder_;
};

}  // namespace deep_yolo_inference
