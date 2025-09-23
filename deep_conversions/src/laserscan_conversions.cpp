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

#include "deep_conversions/laserscan_conversions.hpp"

#include <cstring>
#include <memory>
#include <vector>

namespace deep_ros
{
namespace ros_conversions
{

Tensor from_laserscan(const sensor_msgs::msg::LaserScan & scan, std::shared_ptr<BackendMemoryAllocator> allocator)
{
  size_t num_ranges = scan.ranges.size();
  bool has_intensities = !scan.intensities.empty() && scan.intensities.size() == num_ranges;

  if (has_intensities) {
    std::vector<size_t> shape = {num_ranges, 2};
    Tensor tensor(shape, DataType::FLOAT32, allocator);
    auto * data = tensor.data_as<float>();

    for (size_t i = 0; i < num_ranges; ++i) {
      data[i * 2] = scan.ranges[i];
      data[i * 2 + 1] = scan.intensities[i];
    }
    return tensor;
  } else {
    std::vector<size_t> shape = {num_ranges};
    Tensor tensor(shape, DataType::FLOAT32, allocator);
    if (allocator) {
      allocator->copy_from_host(tensor.data(), scan.ranges.data(), num_ranges * sizeof(float));
    } else {
      std::memcpy(tensor.data(), scan.ranges.data(), num_ranges * sizeof(float));
    }
    return tensor;
  }
}

}  // namespace ros_conversions
}  // namespace deep_ros
