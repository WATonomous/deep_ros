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

#include "deep_core/deep_node_base.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pluginlib/class_loader.hpp>

namespace deep_ros
{

DeepNodeBase::DeepNodeBase(const std::string & node_name, const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode(node_name, options)
, model_loaded_(false)
{
  declare_parameters();

  RCLCPP_INFO(get_logger(), "DeepNodeBase initialized: %s", node_name.c_str());
}

CallbackReturn DeepNodeBase::on_configure(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Configuring node...");

  // Get plugin name from parameters
  std::string plugin_name;
  if (!get_parameter("plugin_name", plugin_name)) {
    RCLCPP_ERROR(get_logger(), "Missing required parameter: plugin_name");
    return CallbackReturn::FAILURE;
  }

  // Load the plugin
  if (!load_plugin(plugin_name)) {
    RCLCPP_ERROR(get_logger(), "Failed to load plugin: %s", plugin_name.c_str());
    return CallbackReturn::FAILURE;
  }

  // Get model path from parameters
  std::string model_path_str;
  if (get_parameter("model_path", model_path_str)) {
    if (!load_model(std::filesystem::path(model_path_str))) {
      RCLCPP_ERROR(get_logger(), "Failed to load model: %s", model_path_str.c_str());
      return CallbackReturn::FAILURE;
    }
  }

  // Call user implementation
  auto result = on_configure_impl(state);
  if (result != CallbackReturn::SUCCESS) {
    RCLCPP_ERROR(get_logger(), "User configure implementation failed");
    return result;
  }

  RCLCPP_INFO(get_logger(), "Node configured successfully");
  return CallbackReturn::SUCCESS;
}

CallbackReturn DeepNodeBase::on_activate(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Activating node...");

  if (!is_plugin_loaded()) {
    RCLCPP_ERROR(get_logger(), "Cannot activate - no plugin loaded");
    return CallbackReturn::FAILURE;
  }

  // Call user implementation
  auto result = on_activate_impl(state);
  if (result != CallbackReturn::SUCCESS) {
    RCLCPP_ERROR(get_logger(), "User activate implementation failed");
    return result;
  }

  RCLCPP_INFO(get_logger(), "Node activated successfully");
  return CallbackReturn::SUCCESS;
}

CallbackReturn DeepNodeBase::on_deactivate(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Deactivating node...");

  // Call user implementation first
  auto result = on_deactivate_impl(state);
  if (result != CallbackReturn::SUCCESS) {
    RCLCPP_WARN(get_logger(), "User deactivate implementation failed, continuing with cleanup");
  }

  RCLCPP_INFO(get_logger(), "Node deactivated successfully");
  return CallbackReturn::SUCCESS;
}

CallbackReturn DeepNodeBase::on_cleanup(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Cleaning up node...");

  // Call user implementation first
  auto result = on_cleanup_impl(state);
  if (result != CallbackReturn::SUCCESS) {
    RCLCPP_WARN(get_logger(), "User cleanup implementation failed, continuing with base cleanup");
  }

  // Cleanup base resources
  unload_model();
  plugin_.reset();
  current_plugin_name_.clear();

  RCLCPP_INFO(get_logger(), "Node cleaned up successfully");
  return CallbackReturn::SUCCESS;
}

CallbackReturn DeepNodeBase::on_shutdown(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Shutting down node...");

  // Call user implementation first
  auto result = on_shutdown_impl(state);
  if (result != CallbackReturn::SUCCESS) {
    RCLCPP_WARN(get_logger(), "User shutdown implementation failed, continuing with base shutdown");
  }

  // Force cleanup
  unload_model();
  plugin_.reset();
  current_plugin_name_.clear();

  RCLCPP_INFO(get_logger(), "Node shut down successfully");
  return CallbackReturn::SUCCESS;
}

bool DeepNodeBase::load_plugin(const std::string & plugin_name)
{
  try {
    plugin_ = load_plugin_library(plugin_name);
    if (!plugin_) {
      RCLCPP_ERROR(get_logger(), "Failed to load plugin library: %s", plugin_name.c_str());
      return false;
    }

    current_plugin_name_ = plugin_name;
    RCLCPP_INFO(get_logger(), "Loaded plugin: %s (backend: %s)", plugin_name.c_str(), plugin_->backend_name().c_str());
    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception loading plugin %s: %s", plugin_name.c_str(), e.what());
    return false;
  }
}

bool DeepNodeBase::load_model(const std::filesystem::path & model_path)
{
  if (!plugin_) {
    RCLCPP_ERROR(get_logger(), "Cannot load model - no plugin loaded");
    return false;
  }

  if (!std::filesystem::exists(model_path)) {
    RCLCPP_ERROR(get_logger(), "Model file does not exist: %s", model_path.c_str());
    return false;
  }

  try {
    if (plugin_->load_model(model_path)) {
      current_model_path_ = model_path;
      model_loaded_ = true;
      RCLCPP_INFO(get_logger(), "Loaded model: %s", model_path.c_str());
      return true;
    } else {
      RCLCPP_ERROR(get_logger(), "Plugin failed to load model: %s", model_path.c_str());
      return false;
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception loading model %s: %s", model_path.c_str(), e.what());
    return false;
  }
}

void DeepNodeBase::unload_model()
{
  if (plugin_ && model_loaded_) {
    try {
      plugin_->unload_model();
      model_loaded_ = false;
      current_model_path_.clear();
      RCLCPP_INFO(get_logger(), "Model unloaded");
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Exception unloading model: %s", e.what());
    }
  }
}

InferenceResult DeepNodeBase::run_inference(std::vector<std::unique_ptr<Tensor>> inputs)
{
  if (!plugin_) {
    return InferenceResult::error(InferenceError::MODEL_NOT_LOADED, "No plugin loaded");
  }

  if (!model_loaded_) {
    return InferenceResult::error(InferenceError::MODEL_NOT_LOADED, "No model loaded");
  }

  try {
    return plugin_->inference(std::move(inputs));
  } catch (const std::exception & e) {
    return InferenceResult::error(InferenceError::BACKEND_ERROR, std::string("Inference failed: ") + e.what());
  }
}

std::string DeepNodeBase::get_backend_name() const
{
  if (plugin_) {
    return plugin_->backend_name();
  }
  return "none";
}

std::vector<std::string> DeepNodeBase::discover_available_plugins()
{
  // TODO(eddy): Implement plugin discovery using pluginlib
  // For now, return empty vector
  return {};
}

std::unique_ptr<InferencePluginInterface> DeepNodeBase::load_plugin_library(const std::string & plugin_name)
{
  // TODO(eddy): Implement plugin loading using pluginlib class_loader
  // For now, return nullptr
  RCLCPP_ERROR(get_logger(), "Plugin loading not yet implemented");
  return nullptr;
}

void DeepNodeBase::declare_parameters()
{
  // Declare common parameters
  declare_parameter("plugin_name", "");
  declare_parameter("model_path", "");
  declare_parameter("input_timeout_ms", 1000);
  declare_parameter("max_batch_size", 1);
}

}  // namespace deep_ros
