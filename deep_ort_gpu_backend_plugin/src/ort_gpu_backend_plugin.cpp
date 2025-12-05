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

#include "deep_ort_gpu_backend_plugin/ort_gpu_backend_plugin.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include <pluginlib/class_list_macros.hpp>

#include "deep_ort_gpu_backend_plugin/ort_gpu_backend_executor.hpp"
#include "deep_ort_gpu_backend_plugin/ort_gpu_memory_allocator.hpp"

namespace deep_ort_gpu_backend
{

OrtGpuBackendPlugin::OrtGpuBackendPlugin()
: device_id_(0)
, execution_provider_("cuda")
{
  // GPU components will be initialized in initialize() after parameters are loaded
}

void OrtGpuBackendPlugin::initialize(rclcpp_lifecycle::LifecycleNode::SharedPtr node)
{
  node_ = node;

  // Declare parameters with defaults
  node_->declare_parameter("Backend.device_id", 0);
  node_->declare_parameter("Backend.execution_provider", "cuda");

  // Read parameters
  device_id_ = node_->get_parameter("Backend.device_id").as_int();
  execution_provider_ = node_->get_parameter("Backend.execution_provider").as_string();

  RCLCPP_INFO(
    node_->get_logger(),
    "Initializing GPU backend with device_id=%d, execution_provider=%s",
    device_id_,
    execution_provider_.c_str());

  // Initialize GPU components with configured parameters
  if (!initialize_gpu_components()) {
    throw std::runtime_error("Failed to initialize GPU backend components");
  }
}

std::string OrtGpuBackendPlugin::backend_name() const
{
  return "onnxruntime_gpu";
}

std::shared_ptr<deep_ros::BackendMemoryAllocator> OrtGpuBackendPlugin::get_allocator() const
{
  return allocator_;
}

std::shared_ptr<deep_ros::BackendInferenceExecutor> OrtGpuBackendPlugin::get_inference_executor() const
{
  return executor_;
}

int OrtGpuBackendPlugin::get_device_id() const
{
  return device_id_;
}

bool OrtGpuBackendPlugin::initialize_gpu_components()
{
  try {
    // Initialize allocator first
    allocator_ = get_ort_gpu_cpu_allocator();
    if (!allocator_) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to get GPU backend CPU allocator");
      return false;
    }

    // Create GPU executor
    executor_ = std::make_shared<OrtGpuBackendExecutor>(device_id_, execution_provider_, node_->get_logger());
    if (!executor_) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to create GPU backend executor");
      return false;
    }

    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to initialize GPU components: %s", e.what());
    executor_.reset();
    allocator_.reset();
    return false;
  }
}

}  // namespace deep_ort_gpu_backend

// Export the plugin class
PLUGINLIB_EXPORT_CLASS(deep_ort_gpu_backend::OrtGpuBackendPlugin, deep_ros::DeepBackendPlugin)
