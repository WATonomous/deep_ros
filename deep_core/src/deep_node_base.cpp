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

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <bondcpp/bond.hpp>

namespace deep_ros
{

DeepNodeBase::DeepNodeBase(const std::string & node_name, const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode(node_name, options)
, model_loaded_(false)
{
  plugin_loader_ =
    std::make_unique<pluginlib::ClassLoader<DeepBackendPlugin>>("deep_core", "deep_ros::DeepBackendPlugin");
  declare_parameters();
}

DeepNodeBase::~DeepNodeBase() = default;

void DeepNodeBase::declare_parameters()
{
  declare_parameter("Backend.plugin", "");
  declare_parameter("model_path", "");

  // Bond parameters
  declare_parameter("Bond.enable", false);
  declare_parameter("Bond.bond_timeout", 4.0);
  declare_parameter("Bond.bond_heartbeat_period", 0.1);
}

CallbackReturn DeepNodeBase::on_configure(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Configuring DeepNodeBase");

  // Setup bond if enabled
  setup_bond();

  std::string backend_plugin = get_parameter("Backend.plugin").as_string();
  if (!backend_plugin.empty()) {
    if (!load_plugin(backend_plugin)) {
      RCLCPP_ERROR(get_logger(), "Failed to load backend plugin: %s", backend_plugin.c_str());
      return CallbackReturn::FAILURE;
    }
  }

  std::string model_path = get_parameter("model_path").as_string();
  if (!model_path.empty()) {
    if (!load_model(model_path)) {
      RCLCPP_ERROR(get_logger(), "Failed to load model: %s", model_path.c_str());
      return CallbackReturn::FAILURE;
    }
  }

  return on_configure_impl(state);
}

CallbackReturn DeepNodeBase::on_activate(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Activating DeepNodeBase");

  parameter_callback_handle_ =
    add_on_set_parameters_callback(std::bind(&DeepNodeBase::on_parameter_change, this, std::placeholders::_1));

  // Start bond if enabled
  if (bond_enabled_ && bond_) {
    bond_->start();
    RCLCPP_INFO(get_logger(), "Bond started");
  }

  return on_activate_impl(state);
}

CallbackReturn DeepNodeBase::on_deactivate(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Deactivating DeepNodeBase");

  if (parameter_callback_handle_) {
    remove_on_set_parameters_callback(parameter_callback_handle_.get());
    parameter_callback_handle_.reset();
  }

  return on_deactivate_impl(state);
}

CallbackReturn DeepNodeBase::on_cleanup(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Cleaning up DeepNodeBase");

  // Stop bond if active
  if (bond_) {
    bond_.reset();
    RCLCPP_INFO(get_logger(), "Bond stopped and cleaned up");
  }

  unload_model();
  plugin_.reset();
  return on_cleanup_impl(state);
}

CallbackReturn DeepNodeBase::on_shutdown(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Shutting down DeepNodeBase");

  if (parameter_callback_handle_) {
    remove_on_set_parameters_callback(parameter_callback_handle_.get());
    parameter_callback_handle_.reset();
  }

  // Stop bond if active
  if (bond_) {
    bond_.reset();
    RCLCPP_INFO(get_logger(), "Bond stopped and cleaned up");
  }

  unload_model();
  plugin_.reset();
  return on_shutdown_impl(state);
}

bool DeepNodeBase::load_plugin(const std::string & plugin_name)
{
  try {
    RCLCPP_INFO(get_logger(), "Loading plugin: %s", plugin_name.c_str());
    plugin_ = plugin_loader_->createUniqueInstance(plugin_name);
    current_plugin_name_ = plugin_name;
    RCLCPP_INFO(get_logger(), "Successfully loaded plugin: %s", plugin_name.c_str());
    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Failed to load plugin %s: %s", plugin_name.c_str(), e.what());
    return false;
  }
}

bool DeepNodeBase::load_model(const std::filesystem::path & model_path)
{
  if (!plugin_) {
    RCLCPP_ERROR(get_logger(), "No plugin loaded - cannot load model");
    return false;
  }

  try {
    RCLCPP_INFO(get_logger(), "Loading model: %s", model_path.c_str());
    auto executor = plugin_->get_inference_executor();
    if (executor) {
      model_loaded_ = executor->load_model(model_path);
    } else {
      model_loaded_ = false;
    }
    if (model_loaded_) {
      current_model_path_ = model_path;
      RCLCPP_INFO(get_logger(), "Successfully loaded model: %s", model_path.c_str());
    } else {
      RCLCPP_ERROR(get_logger(), "Plugin failed to load model: %s", model_path.c_str());
    }
    return model_loaded_;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception loading model %s: %s", model_path.c_str(), e.what());
    return false;
  }
}

void DeepNodeBase::unload_model()
{
  if (plugin_ && model_loaded_) {
    RCLCPP_INFO(get_logger(), "Unloading model");
    auto executor = plugin_->get_inference_executor();
    if (executor) {
      executor->unload_model();
    }
    model_loaded_ = false;
    current_model_path_.clear();
  }
}

Tensor DeepNodeBase::run_inference(const Tensor & inputs)
{
  if (!plugin_) {
    throw std::runtime_error("No plugin loaded");
  }

  if (!model_loaded_) {
    throw std::runtime_error("No model loaded");
  }

  auto executor = plugin_->get_inference_executor();
  if (!executor) {
    throw std::runtime_error("No inference executor available");
  }

  return executor->run_inference(inputs);
}

std::string DeepNodeBase::get_backend_name() const
{
  if (plugin_) {
    return plugin_->backend_name();
  }
  return "none";
}

std::shared_ptr<BackendMemoryAllocator> DeepNodeBase::get_current_allocator() const
{
  if (plugin_) {
    return plugin_->get_allocator();
  }
  return nullptr;
}

std::vector<std::string> DeepNodeBase::discover_available_plugins()
{
  return plugin_loader_->getDeclaredClasses();
}

pluginlib::UniquePtr<DeepBackendPlugin> DeepNodeBase::load_plugin_library(const std::string & plugin_name)
{
  return plugin_loader_->createUniqueInstance(plugin_name);
}

void DeepNodeBase::setup_bond()
{
  bond_enabled_ = get_parameter("Bond.enable").as_bool();
  bond_timeout_ = get_parameter("Bond.bond_timeout").as_double();
  bond_heartbeat_period_ = get_parameter("Bond.bond_heartbeat_period").as_double();

  if (bond_enabled_) {
    RCLCPP_INFO(
      get_logger(), "Setting up bond with timeout: %.1fs, heartbeat: %.1fs", bond_timeout_, bond_heartbeat_period_);

    // Create bond with standard topic name and node name as bond ID
    std::string bond_topic = "/bond";
    bond_ = std::make_unique<bond::Bond>(bond_topic, get_name(), shared_from_this());

    // Configure bond parameters
    bond_->setHeartbeatPeriod(bond_heartbeat_period_);
    bond_->setHeartbeatTimeout(bond_timeout_);

    RCLCPP_INFO(get_logger(), "Bond configured on topic: %s", bond_topic.c_str());
  } else {
    RCLCPP_INFO(get_logger(), "Bond disabled");
  }
}

rcl_interfaces::msg::SetParametersResult DeepNodeBase::on_parameter_change(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  for (const auto & param : parameters) {
    if (param.get_name() == "model_path") {
      // Only allow model changes when node is active for safety
      if (get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
        std::string new_model_path = param.as_string();

        // Reject empty model paths entirely
        if (new_model_path.empty()) {
          RCLCPP_ERROR(get_logger(), "Cannot set empty model path");
          result.successful = false;
          result.reason = "Cannot set empty model path";
        } else if (new_model_path != current_model_path_.string()) {
          RCLCPP_INFO(
            get_logger(),
            "Dynamically changing model from '%s' to '%s'",
            current_model_path_.c_str(),
            new_model_path.c_str());

          // Unload current model
          unload_model();

          // Load new model
          if (!load_model(new_model_path)) {
            RCLCPP_ERROR(get_logger(), "Failed to load new model: %s", new_model_path.c_str());
            result.successful = false;
            result.reason = "Failed to load new model: " + new_model_path;
          } else {
            RCLCPP_INFO(get_logger(), "Successfully loaded new model: %s", new_model_path.c_str());
          }
        }
      } else {
        RCLCPP_WARN(
          get_logger(),
          "Cannot change model_path when node is not active. Current state: %s",
          get_current_state().label().c_str());
        result.successful = false;
        result.reason = "Node must be active to change model_path";
      }
    }
  }

  return result;
}

}  // namespace deep_ros
