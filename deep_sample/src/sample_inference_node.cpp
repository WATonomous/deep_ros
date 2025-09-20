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

#include "deep_sample/sample_inference_node.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <deep_conversions/image_conversions.hpp>
#include <deep_core/types/tensor.hpp>

namespace deep_sample
{

SampleInferenceNode::SampleInferenceNode(const rclcpp::NodeOptions & options)
: DeepNodeBase("sample_inference_node", options)
{
  RCLCPP_INFO(get_logger(), "SampleInferenceNode constructor");
}

deep_ros::CallbackReturn SampleInferenceNode::on_configure_impl(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Configuring SampleInferenceNode");

  // Declare additional parameters
  declare_parameter("input_topic", "/camera/image_raw");
  declare_parameter("output_topic", "/inference/output");

  // Get parameters
  input_topic_ = get_parameter("input_topic").as_string();
  output_topic_ = get_parameter("output_topic").as_string();

  RCLCPP_INFO(get_logger(), "Input topic: %s", input_topic_.c_str());
  RCLCPP_INFO(get_logger(), "Output topic: %s", output_topic_.c_str());

  // Create publisher
  output_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(output_topic_, 10);

  return deep_ros::CallbackReturn::SUCCESS;
}

deep_ros::CallbackReturn SampleInferenceNode::on_activate_impl(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Activating SampleInferenceNode");

  // Check if backend is loaded and model is loaded
  if (!is_plugin_loaded()) {
    RCLCPP_ERROR(get_logger(), "No backend plugin loaded - cannot activate");
    return deep_ros::CallbackReturn::FAILURE;
  }

  if (!is_model_loaded()) {
    RCLCPP_ERROR(get_logger(), "No model loaded - cannot activate");
    return deep_ros::CallbackReturn::FAILURE;
  }

  // Create subscriber
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    input_topic_, 10, std::bind(&SampleInferenceNode::image_callback, this, std::placeholders::_1));

  // Activate publisher
  output_pub_->on_activate();

  RCLCPP_INFO(get_logger(), "SampleInferenceNode activated with backend: %s", get_backend_name().c_str());

  return deep_ros::CallbackReturn::SUCCESS;
}

deep_ros::CallbackReturn SampleInferenceNode::on_deactivate_impl(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Deactivating SampleInferenceNode");

  // Reset subscriber
  image_sub_.reset();

  // Deactivate publisher
  output_pub_->on_deactivate();

  return deep_ros::CallbackReturn::SUCCESS;
}

deep_ros::CallbackReturn SampleInferenceNode::on_cleanup_impl(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Cleaning up SampleInferenceNode");

  // Reset all resources
  image_sub_.reset();
  output_pub_.reset();

  return deep_ros::CallbackReturn::SUCCESS;
}

void SampleInferenceNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  try {
    RCLCPP_DEBUG(get_logger(), "Received image: %dx%d, encoding: %s", msg->width, msg->height, msg->encoding.c_str());

    // Get allocator from the current backend plugin
    auto allocator = get_current_allocator();
    if (!allocator) {
      RCLCPP_ERROR(get_logger(), "No allocator available from backend plugin");
      return;
    }

    // Convert image to tensor using deep_conversions
    auto input_tensor = deep_ros::ros_conversions::from_image(*msg, allocator);

    // Run inference
    auto output_tensor = run_inference(std::move(input_tensor));

    // Convert output to message and publish
    auto output_msg = tensor_to_output(output_tensor);
    output_msg.layout.dim.resize(output_tensor.rank());
    for (size_t i = 0; i < output_tensor.rank(); ++i) {
      output_msg.layout.dim[i].label = "dim_" + std::to_string(i);
      output_msg.layout.dim[i].size = output_tensor.shape()[i];
      output_msg.layout.dim[i].stride = 1;
    }

    output_pub_->publish(output_msg);

    RCLCPP_DEBUG(get_logger(), "Published inference output with %zu elements", output_msg.data.size());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Inference failed: %s", e.what());
  }
}

std_msgs::msg::Float32MultiArray SampleInferenceNode::tensor_to_output(const deep_ros::Tensor & tensor)
{
  std_msgs::msg::Float32MultiArray output_msg;

  // Copy tensor data to output message
  const float * data = tensor.data_as<float>();
  output_msg.data.resize(tensor.size());

  for (size_t i = 0; i < tensor.size(); ++i) {
    output_msg.data[i] = data[i];
  }

  return output_msg;
}

}  // namespace deep_sample

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(deep_sample::SampleInferenceNode)
