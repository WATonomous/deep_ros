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

#include <sensor_msgs/msg/laser_scan.hpp>

#include "deep_core/plugin_interfaces/backend_memory_allocator.hpp"
#include "deep_core/types/tensor.hpp"

namespace deep_ros
{
namespace ros_conversions
{

/**
 * @brief Convert sensor_msgs::msg::LaserScan to Tensor
 *
 * Converts a ROS LaserScan message to a tensor format. If intensity data is present
 * and matches the number of range readings, the output tensor will have shape
 * [num_ranges, 2] with ranges and intensities. Otherwise, it will have shape
 * [num_ranges] with only range data.
 *
 * @param scan ROS LaserScan message
 * @param allocator Memory allocator to use (uses CPU allocator if nullptr)
 * @return Tensor with shape [num_ranges] or [num_ranges, 2] if intensities present
 */
Tensor from_laserscan(
  const sensor_msgs::msg::LaserScan & scan, std::shared_ptr<BackendMemoryAllocator> allocator = nullptr);

}  // namespace ros_conversions
}  // namespace deep_ros
