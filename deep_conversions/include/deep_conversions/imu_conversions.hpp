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

#include <sensor_msgs/msg/imu.hpp>

#include "deep_core/plugin_interfaces/backend_memory_allocator.hpp"
#include "deep_core/types/tensor.hpp"

namespace deep_ros
{
namespace ros_conversions
{

/**
 * @brief Convert sensor_msgs::msg::Imu to Tensor
 *
 * Converts a ROS IMU message to a tensor format containing orientation,
 * linear acceleration, and angular velocity data. The output tensor has
 * shape [10] with the following layout:
 * - [0-3]: orientation quaternion (x, y, z, w)
 * - [4-6]: linear acceleration (x, y, z)
 * - [7-9]: angular velocity (x, y, z)
 *
 * @param imu ROS IMU message
 * @param allocator Memory allocator to use (uses CPU allocator if nullptr)
 * @return Tensor with shape [10] containing [qx,qy,qz,qw,ax,ay,az,gx,gy,gz]
 */
Tensor from_imu(const sensor_msgs::msg::Imu & imu, std::shared_ptr<BackendMemoryAllocator> allocator = nullptr);

}  // namespace ros_conversions
}  // namespace deep_ros
