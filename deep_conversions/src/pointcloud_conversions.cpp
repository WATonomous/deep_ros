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

#include "deep_conversions/pointcloud_conversions.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace deep_ros
{
namespace ros_conversions
{

Tensor from_pointcloud2(const sensor_msgs::msg::PointCloud2 & cloud, std::shared_ptr<BackendMemoryAllocator> allocator)
{
  if (cloud.fields.empty()) {
    throw std::invalid_argument("PointCloud2 has no fields");
  }

  size_t num_points = cloud.width * cloud.height;
  size_t num_fields = cloud.fields.size();

  // For simplicity, convert all to float32
  std::vector<size_t> shape = {num_points, num_fields};
  Tensor tensor(shape, DataType::FLOAT32, allocator);
  auto * data = tensor.data_as<float>();

  // Extract field data
  for (size_t i = 0; i < num_points; ++i) {
    for (size_t j = 0; j < num_fields; ++j) {
      const auto & field = cloud.fields[j];
      size_t offset = i * cloud.point_step + field.offset;

      // Convert based on datatype
      float value = 0.0f;
      if (field.datatype == sensor_msgs::msg::PointField::FLOAT32) {
        std::memcpy(&value, &cloud.data[offset], sizeof(float));
      } else if (field.datatype == sensor_msgs::msg::PointField::FLOAT64) {
        double temp;
        std::memcpy(&temp, &cloud.data[offset], sizeof(double));
        value = static_cast<float>(temp);
      } else if (field.datatype == sensor_msgs::msg::PointField::INT32) {
        int32_t temp;
        std::memcpy(&temp, &cloud.data[offset], sizeof(int32_t));
        value = static_cast<float>(temp);
      } else if (field.datatype == sensor_msgs::msg::PointField::UINT32) {
        uint32_t temp;
        std::memcpy(&temp, &cloud.data[offset], sizeof(uint32_t));
        value = static_cast<float>(temp);
      }

      data[i * num_fields + j] = value;
    }
  }

  return tensor;
}

}  // namespace ros_conversions
}  // namespace deep_ros
