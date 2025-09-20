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

#include <deep_conversions/image_conversions.hpp>
#include <deep_core/deep_node_base.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

namespace deep_sample
{

/**
 * @brief Sample inference node demonstrating deep_core usage
 *
 * This node subscribes to image messages, runs inference using a loaded model,
 * and publishes the results. It demonstrates how to use DeepNodeBase with
 * the ONNX Runtime backend plugin.
 */
class SampleInferenceNode : public deep_ros::DeepNodeBase
{
public:
  /**
   * @brief Constructor
   * @param options ROS 2 node options
   */
  explicit SampleInferenceNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  /**
   * @brief Destructor
   */
  ~SampleInferenceNode() override = default;

protected:
  /**
   * @brief Configure lifecycle callback
   * @param state Current lifecycle state
   * @return Callback return status
   */
  deep_ros::CallbackReturn on_configure_impl(const rclcpp_lifecycle::State & state) override;

  /**
   * @brief Activate lifecycle callback
   * @param state Current lifecycle state
   * @return Callback return status
   */
  deep_ros::CallbackReturn on_activate_impl(const rclcpp_lifecycle::State & state) override;

  /**
   * @brief Deactivate lifecycle callback
   * @param state Current lifecycle state
   * @return Callback return status
   */
  deep_ros::CallbackReturn on_deactivate_impl(const rclcpp_lifecycle::State & state) override;

  /**
   * @brief Cleanup lifecycle callback
   * @param state Current lifecycle state
   * @return Callback return status
   */
  deep_ros::CallbackReturn on_cleanup_impl(const rclcpp_lifecycle::State & state) override;

private:
  /**
   * @brief Image callback for inference
   * @param msg Input image message
   */
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

  /**
   * @brief Convert tensor to output message
   * @param tensor Output tensor from inference
   * @return Float32MultiArray message
   */
  std_msgs::msg::Float32MultiArray tensor_to_output(const deep_ros::Tensor & tensor);

  // ROS 2 subscribers and publishers
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float32MultiArray>::SharedPtr output_pub_;

  // Parameters
  std::string input_topic_;
  std::string output_topic_;
};

}  // namespace deep_sample
