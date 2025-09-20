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

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <pluginlib/class_list_macros.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include "deep_core/plugin_interfaces/backend_inference_executor.hpp"
#include "deep_core/plugin_interfaces/backend_memory_allocator.hpp"
#include "deep_core/plugin_interfaces/deep_backend_plugin.hpp"
#include "deep_core/types/tensor.hpp"

namespace deep_ros
{

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

// Generic ROS lifecycle node that loads and manages plugins
class DeepNodeBase : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit DeepNodeBase(const std::string & node_name, const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  virtual ~DeepNodeBase() = default;

protected:
  // Users override these for custom behavior
  virtual CallbackReturn on_configure_impl(const rclcpp_lifecycle::State & state)
  {
    return CallbackReturn::SUCCESS;
  }

  virtual CallbackReturn on_activate_impl(const rclcpp_lifecycle::State & state)
  {
    return CallbackReturn::SUCCESS;
  }

  virtual CallbackReturn on_deactivate_impl(const rclcpp_lifecycle::State & state)
  {
    return CallbackReturn::SUCCESS;
  }

  virtual CallbackReturn on_cleanup_impl(const rclcpp_lifecycle::State & state)
  {
    return CallbackReturn::SUCCESS;
  }

  virtual CallbackReturn on_shutdown_impl(const rclcpp_lifecycle::State & state)
  {
    return CallbackReturn::SUCCESS;
  }

  // Plugin management available to users
  bool load_plugin(const std::string & plugin_name);
  bool load_model(const std::filesystem::path & model_path);
  void unload_model();
  Tensor run_inference(Tensor inputs);

  // Get current allocator from loaded plugin
  std::shared_ptr<BackendMemoryAllocator> get_current_allocator() const;

  // Plugin status
  bool is_plugin_loaded() const
  {
    return plugin_ != nullptr;
  }

  bool is_model_loaded() const
  {
    return model_loaded_;
  }

  std::string get_backend_name() const;

private:
  // Final lifecycle callbacks - base handles backend, then calls user impl
  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) final;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) final;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) final;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) final;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) final;

  // Plugin discovery and loading
  std::vector<std::string> discover_available_plugins();
  pluginlib::UniquePtr<DeepBackendPlugin> load_plugin_library(const std::string & plugin_name);

  // Plugin loader
  std::unique_ptr<pluginlib::ClassLoader<DeepBackendPlugin>> plugin_loader_;

  // State
  pluginlib::UniquePtr<DeepBackendPlugin> plugin_;
  bool model_loaded_;
  std::string current_plugin_name_;
  std::filesystem::path current_model_path_;

  // ROS parameters
  void declare_parameters();
};

}  // namespace deep_ros
