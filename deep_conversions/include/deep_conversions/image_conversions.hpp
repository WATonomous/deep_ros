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

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "deep_core/plugin_interfaces/backend_memory_allocator.hpp"
#include "deep_core/types/tensor.hpp"

namespace deep_ros
{
namespace ros_conversions
{

/**
 * @brief Image encoding information
 */
struct ImageEncoding
{
  DataType dtype;
  size_t channels;
  size_t bytes_per_channel;
};

/**
 * @brief Get encoding information from ROS image encoding string
 * @param encoding ROS image encoding string (e.g., "rgb8", "bgr8", "mono8")
 * @return ImageEncoding struct with dtype, channels, and bytes_per_channel
 * @throws std::runtime_error if encoding is unsupported
 */
ImageEncoding get_image_encoding_info(const std::string & encoding);

/**
 * @brief Convert sensor_msgs::msg::Image to Tensor
 * @param image ROS Image message
 * @param allocator Memory allocator to use (uses CPU allocator if nullptr)
 * @return Tensor with shape [1, height, width, channels] or [1, height, width]
 * @throws std::runtime_error if image dimensions are invalid or data size mismatches
 */
Tensor from_image(const sensor_msgs::msg::Image & image, std::shared_ptr<BackendMemoryAllocator> allocator = nullptr);

/**
 * @brief Convert vector of sensor_msgs::msg::Image to batched Tensor
 * @param images Vector of ROS Image messages
 * @param allocator Memory allocator to use (uses CPU allocator if nullptr)
 * @return Tensor with shape [batch_size, height, width, channels] or [batch_size, height, width]
 * @throws std::invalid_argument if batch is empty or images have mismatched dimensions/encodings
 */
Tensor from_image(
  const std::vector<sensor_msgs::msg::Image> & images, std::shared_ptr<BackendMemoryAllocator> allocator = nullptr);

/**
 * @brief Convert Tensor to sensor_msgs::msg::Image (single image from batch)
 * @param tensor Tensor with shape [1, height, width] or [1, height, width, channels]
 * @param image Output ROS Image message
 * @param encoding Image encoding (must match tensor dtype and shape)
 * @param header Optional header to set timestamp and frame_id
 * @throws std::invalid_argument if tensor shape/dtype doesn't match encoding
 */
void to_image(
  const Tensor & tensor,
  sensor_msgs::msg::Image & image,
  const std::string & encoding,
  const std_msgs::msg::Header & header = std_msgs::msg::Header());

/**
 * @brief Convert batched Tensor to vector of sensor_msgs::msg::Image
 * @param tensor Tensor with shape [batch_size, height, width] or [batch_size, height, width, channels]
 * @param images Output vector of ROS Image messages
 * @param encoding Image encoding (must match tensor dtype and shape)
 * @param header Optional header to set timestamp and frame_id
 * @throws std::invalid_argument if tensor shape/dtype doesn't match encoding or batch size is 0
 */
void to_image(
  const Tensor & tensor,
  std::vector<sensor_msgs::msg::Image> & images,
  const std::string & encoding,
  const std_msgs::msg::Header & header = std_msgs::msg::Header());

}  // namespace ros_conversions
}  // namespace deep_ros
