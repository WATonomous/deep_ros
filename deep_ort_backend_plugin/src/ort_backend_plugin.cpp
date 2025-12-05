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

#include "deep_ort_backend_plugin/ort_backend_plugin.hpp"

#include <memory>
#include <string>

#include <pluginlib/class_list_macros.hpp>

#include "deep_ort_backend_plugin/ort_backend_executor.hpp"
#include "deep_ort_backend_plugin/ort_cpu_memory_allocator.hpp"

namespace deep_ort_backend
{

OrtBackendPlugin::OrtBackendPlugin()
: allocator_(get_ort_cpu_allocator())
, executor_(std::make_shared<OrtBackendExecutor>())
{}

void OrtBackendPlugin::initialize(rclcpp_lifecycle::LifecycleNode::SharedPtr node)
{
  node_ = node;
}

std::string OrtBackendPlugin::backend_name() const
{
  return "onnxruntime_cpu";
}

std::shared_ptr<deep_ros::BackendMemoryAllocator> OrtBackendPlugin::get_allocator() const
{
  return allocator_;
}

std::shared_ptr<deep_ros::BackendInferenceExecutor> OrtBackendPlugin::get_inference_executor() const
{
  return executor_;
}

}  // namespace deep_ort_backend

// Export the plugin class
PLUGINLIB_EXPORT_CLASS(deep_ort_backend::OrtBackendPlugin, deep_ros::DeepBackendPlugin)
