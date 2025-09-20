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

/**
 * @file test_minimal_image.cpp
 * @brief Test that image conversions work with minimal dependencies
 *
 * This test demonstrates that you can use image conversions without
 * depending on pointcloud, laserscan, or imu message types.
 */

#include <memory>

// Only include what we need - no pointcloud/laserscan/imu dependencies
#include <deep_conversions/image_conversions.hpp>
#include <sensor_msgs/msg/image.hpp>

int main()
{
  try {
    // Test that we can get encoding info
    auto encoding_info = deep_ros::ros_conversions::get_image_encoding_info("rgb8");

    // Verify the encoding info is correct
    if (
      encoding_info.dtype != deep_ros::DataType::UINT8 || encoding_info.channels != 3 ||
      encoding_info.bytes_per_channel != 1)
    {
      return 1;
    }

    // Test creating a simple image message
    sensor_msgs::msg::Image test_image;
    test_image.height = 10;
    test_image.width = 10;
    test_image.encoding = "rgb8";
    test_image.step = 30;  // 10 * 3
    test_image.data.resize(300, 128);  // Fill with gray

    // This would require an allocator to actually convert
    // but we can at least verify the function exists and compiles

    return 0;  // Success
  } catch (const std::exception & e) {
    return 1;  // Failure
  }
}
