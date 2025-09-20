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

#include <cstring>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "deep_tensor/memory_allocator.hpp"
#include "deep_tensor/tensor.hpp"

namespace deep_ros
{
namespace ros_conversions
{

struct ImageEncoding
{
  DataType dtype;
  size_t channels;
  size_t bytes_per_channel;
};

/**
 * @brief Get encoding information from ROS image encoding string
 */
ImageEncoding get_image_encoding_info(const std::string & encoding);

/**
 * @brief Convert sensor_msgs::msg::Image to Tensor
 * @param image ROS Image message
 * @param allocator Memory allocator to use (uses CPU allocator if nullptr)
 * @return Tensor with shape [1, height, width, channels] or [1, height, width]
 */
Tensor from_image(const sensor_msgs::msg::Image & image, std::shared_ptr<MemoryAllocator> allocator = nullptr);

/**
 * @brief Convert vector of sensor_msgs::msg::Image to batched Tensor
 * @param images Vector of ROS Image messages
 * @param allocator Memory allocator to use (uses CPU allocator if nullptr)
 * @return Tensor with shape [batch_size, height, width, channels] or [batch_size, height, width]
 */
Tensor from_image(
  const std::vector<sensor_msgs::msg::Image> & images, std::shared_ptr<MemoryAllocator> allocator = nullptr);

/**
 * @brief Convert Tensor to sensor_msgs::msg::Image (single image from batch)
 * @param tensor Tensor with shape [1, height, width] or [1, height, width, channels]
 * @param image Output ROS Image message
 * @param encoding Image encoding (must match tensor dtype and shape)
 * @param header Optional header to set timestamp
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
 * @param header Optional header to set timestamp
 */
void to_image(
  const Tensor & tensor,
  std::vector<sensor_msgs::msg::Image> & images,
  const std::string & encoding,
  const std_msgs::msg::Header & header = std_msgs::msg::Header());

/**
 * @brief Convert sensor_msgs::msg::PointCloud2 to Tensor
 * @param cloud ROS PointCloud2 message
 * @param allocator Memory allocator to use (uses CPU allocator if nullptr)
 * @return Tensor with shape [num_points, num_fields]
 */
Tensor from_pointcloud2(
  const sensor_msgs::msg::PointCloud2 & cloud, std::shared_ptr<MemoryAllocator> allocator = nullptr);

/**
 * @brief Convert sensor_msgs::msg::LaserScan to Tensor
 * @param scan ROS LaserScan message
 * @param allocator Memory allocator to use (uses CPU allocator if nullptr)
 * @return Tensor with shape [num_ranges] or [num_ranges, 2] if intensities present
 */
Tensor from_laserscan(const sensor_msgs::msg::LaserScan & scan, std::shared_ptr<MemoryAllocator> allocator = nullptr);

/**
 * @brief Convert sensor_msgs::msg::Imu to Tensor
 * @param imu ROS IMU message
 * @param allocator Memory allocator to use (uses CPU allocator if nullptr)
 * @return Tensor with shape [10] containing [qx,qy,qz,qw,ax,ay,az,gx,gy,gz]
 */
Tensor from_imu(const sensor_msgs::msg::Imu & imu, std::shared_ptr<MemoryAllocator> allocator = nullptr);

}  // namespace ros_conversions
}  // namespace deep_ros
