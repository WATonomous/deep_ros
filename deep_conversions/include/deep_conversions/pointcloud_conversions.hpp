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

#include <sensor_msgs/msg/point_cloud2.hpp>

#include "deep_core/plugin_interfaces/backend_memory_allocator.hpp"
#include "deep_core/types/tensor.hpp"

namespace deep_ros
{
namespace ros_conversions
{

/**
 * @brief Convert sensor_msgs::msg::PointCloud2 to Tensor
 *
 * Converts a ROS PointCloud2 message to a tensor format suitable for deep learning.
 * All field data types are converted to FLOAT32 for consistency.
 *
 * @param cloud ROS PointCloud2 message
 * @param allocator Memory allocator to use (uses CPU allocator if nullptr)
 * @return Tensor with shape [num_points, num_fields] containing point data
 * @throws std::invalid_argument if pointcloud has no fields
 */
Tensor from_pointcloud2(
  const sensor_msgs::msg::PointCloud2 & cloud, std::shared_ptr<BackendMemoryAllocator> allocator = nullptr);

}  // namespace ros_conversions
}  // namespace deep_ros
