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

#include <bondcpp/bond.hpp>
#include <lifecycle_msgs/msg/state.hpp>
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

/**
 * @brief Base class for deep learning inference nodes in ROS 2
 *
 * DeepNodeBase provides a lifecycle-managed ROS 2 node that can load and manage
 * deep learning backend plugins. It handles the plugin discovery, loading, model
 * management, and provides a simple interface for running inference.
 */
class DeepNodeBase : public rclcpp_lifecycle::LifecycleNode
{
public:
  /**
   * @brief Construct a new DeepNodeBase - initialize base parameters
   * @param node_name Name of the ROS 2 node
   * @param options ROS 2 node options
   */
  explicit DeepNodeBase(const std::string & node_name, const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  /**
   * @brief Destructor
   */
  virtual ~DeepNodeBase() = default;

protected:
  /**
   * @brief User-implemented configuration callback
   * @param state Current lifecycle state
   * @return Callback return status
   */
  virtual CallbackReturn on_configure_impl(const rclcpp_lifecycle::State & state)
  {
    return CallbackReturn::SUCCESS;
  }

  /**
   * @brief User-implemented activation callback
   * @param state Current lifecycle state
   * @return Callback return status
   */
  virtual CallbackReturn on_activate_impl(const rclcpp_lifecycle::State & state)
  {
    return CallbackReturn::SUCCESS;
  }

  /**
   * @brief User-implemented deactivation callback
   * @param state Current lifecycle state
   * @return Callback return status
   */
  virtual CallbackReturn on_deactivate_impl(const rclcpp_lifecycle::State & state)
  {
    return CallbackReturn::SUCCESS;
  }

  /**
   * @brief User-implemented cleanup callback
   * @param state Current lifecycle state
   * @return Callback return status
   */
  virtual CallbackReturn on_cleanup_impl(const rclcpp_lifecycle::State & state)
  {
    return CallbackReturn::SUCCESS;
  }

  /**
   * @brief User-implemented shutdown callback
   * @param state Current lifecycle state
   * @return Callback return status
   */
  virtual CallbackReturn on_shutdown_impl(const rclcpp_lifecycle::State & state)
  {
    return CallbackReturn::SUCCESS;
  }

  /**
   * @brief Load a backend plugin by name
   * @param plugin_name Name of the plugin to load
   * @return True if successful, false otherwise
   */
  bool load_plugin(const std::string & plugin_name);

  /**
   * @brief Load a model file for inference
   * @param model_path Path to the model file
   * @return True if successful, false otherwise
   */
  bool load_model(const std::filesystem::path & model_path);

  /**
   * @brief Unload the currently loaded model
   */
  void unload_model();

  /**
   * @brief Run inference on input tensor
   * @param inputs Input tensor for inference
   * @return Output tensor from inference
   */
  Tensor run_inference(Tensor inputs);

  /**
   * @brief Check if a backend plugin is loaded
   * @return True if plugin is loaded, false otherwise
   */
  bool is_plugin_loaded() const
  {
    return plugin_ != nullptr;
  }

  /**
   * @brief Check if a model is loaded and ready for inference
   * @return True if model is loaded, false otherwise
   */
  bool is_model_loaded() const
  {
    return model_loaded_;
  }

  /**
   * @brief Get the name of the currently loaded backend
   * @return Backend name string, or empty string if no plugin loaded
   */
  std::string get_backend_name() const;

  /**
   * @brief Get the memory allocator from the currently loaded plugin
   * @return Shared pointer to memory allocator, or nullptr if no plugin loaded
   */
  std::shared_ptr<BackendMemoryAllocator> get_current_allocator() const;

private:
  /**
   * @brief Configure lifecycle callback - retrieve parameter values,
   * loads plugin and model, then calls user implementation
   * @param state Current lifecycle state
   * @return Callback return status
   */
  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) final;

  /**
   * @brief Activate lifecycle callback - starts bond if enabled, then calls user implementation
   * @param state Current lifecycle state
   * @return Callback return status
   */
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) final;

  /**
   * @brief Deactivate lifecycle callback - calls user implementation
   * @param state Current lifecycle state
   * @return Callback return status
   */
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) final;

  /**
   * @brief Cleanup lifecycle callback - unloads model/plugin, stops bond, then calls user implementation
   * @param state Current lifecycle state
   * @return Callback return status
   */
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) final;

  /**
   * @brief Shutdown lifecycle callback - unloads model/plugin, stops bond, then calls user implementation
   * @param state Current lifecycle state
   * @return Callback return status
   */
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) final;

  /**
   * @brief Discover available backend plugins using pluginlib
   * @return Vector of plugin class names
   */
  std::vector<std::string> discover_available_plugins();

  /**
   * @brief Load a specific backend plugin library
   * @param plugin_name Name of the plugin class to load
   * @return Unique pointer to the loaded plugin instance
   */
  pluginlib::UniquePtr<DeepBackendPlugin> load_plugin_library(const std::string & plugin_name);

  // Plugin loader
  std::unique_ptr<pluginlib::ClassLoader<DeepBackendPlugin>> plugin_loader_;

  // State
  pluginlib::UniquePtr<DeepBackendPlugin> plugin_;
  bool model_loaded_;
  std::string current_plugin_name_;
  std::filesystem::path current_model_path_;

  // Bond support
  std::unique_ptr<bond::Bond> bond_;
  bool bond_enabled_;
  double bond_timeout_;
  double bond_heartbeat_period_;

  // ROS parameters
  void declare_parameters();
  void setup_bond();
  rcl_interfaces::msg::SetParametersResult on_parameter_change(const std::vector<rclcpp::Parameter> & parameters);

  // Parameter callback for dynamic reconfiguration
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
};

}  // namespace deep_ros
