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

#include "deep_conversions/imu_conversions.hpp"

#include <memory>
#include <vector>

namespace deep_ros
{
namespace ros_conversions
{

Tensor from_imu(const sensor_msgs::msg::Imu & imu, std::shared_ptr<BackendMemoryAllocator> allocator)
{
  std::vector<size_t> shape = {10};
  Tensor tensor(shape, DataType::FLOAT32, allocator);
  auto * data = tensor.data_as<float>();

  // Orientation quaternion
  data[0] = static_cast<float>(imu.orientation.x);
  data[1] = static_cast<float>(imu.orientation.y);
  data[2] = static_cast<float>(imu.orientation.z);
  data[3] = static_cast<float>(imu.orientation.w);

  // Linear acceleration
  data[4] = static_cast<float>(imu.linear_acceleration.x);
  data[5] = static_cast<float>(imu.linear_acceleration.y);
  data[6] = static_cast<float>(imu.linear_acceleration.z);

  // Angular velocity
  data[7] = static_cast<float>(imu.angular_velocity.x);
  data[8] = static_cast<float>(imu.angular_velocity.y);
  data[9] = static_cast<float>(imu.angular_velocity.z);

  return tensor;
}

}  // namespace ros_conversions
}  // namespace deep_ros
