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

#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>
#include <deep_conversions/image_conversions.hpp>
#include <deep_conversions/imu_conversions.hpp>
#include <deep_conversions/laserscan_conversions.hpp>
#include <deep_conversions/pointcloud_conversions.hpp>
#include <deep_test/deep_test.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace deep_ros
{
namespace ros_conversions
{
namespace test
{

TEST_CASE("Image encoding info parsing", "[conversions][image]")
{
  SECTION("RGB8 encoding")
  {
    auto info = get_image_encoding_info("rgb8");
    REQUIRE(info.dtype == DataType::UINT8);
    REQUIRE(info.channels == 3);
    REQUIRE(info.bytes_per_channel == 1);
  }

  SECTION("BGR8 encoding")
  {
    auto info = get_image_encoding_info("bgr8");
    REQUIRE(info.dtype == DataType::UINT8);
    REQUIRE(info.channels == 3);
    REQUIRE(info.bytes_per_channel == 1);
  }

  SECTION("MONO8 encoding")
  {
    auto info = get_image_encoding_info("mono8");
    REQUIRE(info.dtype == DataType::UINT8);
    REQUIRE(info.channels == 1);
    REQUIRE(info.bytes_per_channel == 1);
  }

  SECTION("RGBA8 encoding")
  {
    auto info = get_image_encoding_info("rgba8");
    REQUIRE(info.dtype == DataType::UINT8);
    REQUIRE(info.channels == 4);
    REQUIRE(info.bytes_per_channel == 1);
  }

  SECTION("MONO16 encoding")
  {
    auto info = get_image_encoding_info("mono16");
    REQUIRE(info.dtype == DataType::UINT16);
    REQUIRE(info.channels == 1);
    REQUIRE(info.bytes_per_channel == 2);
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "Mock Backend SECTION Test", "[conversions][mock]")
{
  auto allocator = getAllocator();
  REQUIRE(allocator != nullptr);

  SECTION("First section")
  {
    // Just allocate some memory
    std::vector<size_t> shape{2, 2};
    Tensor tensor(shape, DataType::FLOAT32, allocator);
    REQUIRE(tensor.data() != nullptr);
  }

  SECTION("Second section")
  {
    // Test conversion function
    sensor_msgs::msg::Image ros_image;
    ros_image.height = 10;
    ros_image.width = 10;
    ros_image.encoding = "rgb8";
    ros_image.step = 10 * 3;
    ros_image.data.resize(10 * 10 * 3, 128);

    auto tensor = from_image(ros_image, allocator);
    REQUIRE(tensor.data() != nullptr);
    REQUIRE(allocator->allocated_bytes() > 0);
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "Image conversion from ROS to Tensor", "[conversions][image]")
{
  auto allocator = getAllocator();
  REQUIRE(allocator != nullptr);

  SECTION("RGB8 image conversion")
  {
    sensor_msgs::msg::Image ros_image;
    ros_image.header.stamp.sec = 123;
    ros_image.header.frame_id = "camera_frame";
    ros_image.height = 100;
    ros_image.width = 100;
    ros_image.encoding = "rgb8";
    ros_image.is_bigendian = false;
    ros_image.step = 100 * 3;  // width * channels
    ros_image.data.resize(100 * 100 * 3);

    // Fill with test pattern
    for (size_t i = 0; i < ros_image.data.size(); ++i) {
      ros_image.data[i] = static_cast<uint8_t>(i % 256);
    }

    auto tensor = from_image(ros_image, allocator);

    REQUIRE(tensor.shape().size() == 4);  // [1, height, width, channels] or similar
    REQUIRE(tensor.dtype() == DataType::UINT8);
    REQUIRE(tensor.data() != nullptr);
    REQUIRE(allocator->allocated_bytes() > 0);
  }

  SECTION("Grayscale image conversion")
  {
    sensor_msgs::msg::Image ros_image;
    ros_image.height = 240;
    ros_image.width = 320;
    ros_image.encoding = "mono8";
    ros_image.step = 320;
    ros_image.data.resize(240 * 320);

    auto tensor = from_image(ros_image, allocator);

    REQUIRE(tensor.shape().size() >= 3);  // At least [1, height, width]
    REQUIRE(tensor.dtype() == DataType::UINT8);
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "Batch image conversion", "[conversions][image][batch]")
{
  auto allocator = getAllocator();

  SECTION("Multiple RGB8 images")
  {
    std::vector<sensor_msgs::msg::Image> images;

    for (int i = 0; i < 3; ++i) {
      sensor_msgs::msg::Image img;
      img.height = 100;
      img.width = 100;
      img.encoding = "rgb8";
      img.step = 100 * 3;
      img.data.resize(100 * 100 * 3, static_cast<uint8_t>(i * 50));
      images.push_back(img);
    }

    auto batch_tensor = from_image(images, allocator);

    // Should have batch dimension
    REQUIRE(batch_tensor.shape().size() == 4);
    REQUIRE(batch_tensor.shape()[0] == 3);  // Batch size
    REQUIRE(batch_tensor.dtype() == DataType::UINT8);
  }
}

TEST_CASE_METHOD(
  deep_ros::test::MockBackendFixture, "Image conversion from Tensor to ROS", "[conversions][image][output]")
{
  auto allocator = getAllocator();

  SECTION("Tensor to single RGB8 image")
  {
    // Create a test tensor
    std::vector<size_t> shape{1, 100, 100, 3};  // [1, height, width, channels]
    Tensor tensor(shape, DataType::UINT8, allocator);

    // Fill with test data
    uint8_t * data = static_cast<uint8_t *>(tensor.data());
    for (size_t i = 0; i < tensor.size(); ++i) {
      data[i] = static_cast<uint8_t>(i % 256);
    }

    sensor_msgs::msg::Image output_image;
    std_msgs::msg::Header header;
    header.stamp.sec = 456;
    header.frame_id = "test_frame";

    to_image(tensor, output_image, "rgb8", header);

    REQUIRE(output_image.height == 100);
    REQUIRE(output_image.width == 100);
    REQUIRE(output_image.encoding == "rgb8");
    REQUIRE(output_image.header.stamp.sec == 456);
    REQUIRE(output_image.header.frame_id == "test_frame");
    REQUIRE(output_image.data.size() == 100 * 100 * 3);
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "PointCloud2 conversion", "[conversions][pointcloud]")
{
  auto allocator = getAllocator();

  SECTION("Basic pointcloud conversion")
  {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp.sec = 789;
    cloud.header.frame_id = "lidar_frame";
    cloud.height = 1;  // Unorganized cloud
    cloud.width = 1000;  // 1000 points
    cloud.is_bigendian = false;
    cloud.point_step = 16;  // 4 fields * 4 bytes each
    cloud.row_step = cloud.point_step * cloud.width;

    // Define fields: x, y, z, intensity
    sensor_msgs::msg::PointField field_x, field_y, field_z, field_i;

    field_x.name = "x";
    field_x.offset = 0;
    field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_x.count = 1;

    field_y.name = "y";
    field_y.offset = 4;
    field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_y.count = 1;

    field_z.name = "z";
    field_z.offset = 8;
    field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_z.count = 1;

    field_i.name = "intensity";
    field_i.offset = 12;
    field_i.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_i.count = 1;

    cloud.fields = {field_x, field_y, field_z, field_i};
    cloud.data.resize(cloud.row_step);

    // Fill with test data
    float * float_data = reinterpret_cast<float *>(cloud.data.data());
    for (size_t i = 0; i < cloud.width; ++i) {
      float_data[i * 4 + 0] = static_cast<float>(i);  // x
      float_data[i * 4 + 1] = static_cast<float>(i * 2);  // y
      float_data[i * 4 + 2] = static_cast<float>(i * 3);  // z
      float_data[i * 4 + 3] = static_cast<float>(i * 4);  // intensity
    }

    auto tensor = from_pointcloud2(cloud, allocator);

    REQUIRE(tensor.shape().size() == 2);  // [num_points, num_fields]
    REQUIRE(tensor.shape()[0] == 1000);  // Number of points
    REQUIRE(tensor.shape()[1] == 4);  // Number of fields
    REQUIRE(tensor.dtype() == DataType::FLOAT32);
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "LaserScan conversion", "[conversions][laserscan]")
{
  auto allocator = getAllocator();

  SECTION("LaserScan with ranges only")
  {
    sensor_msgs::msg::LaserScan scan;
    scan.header.stamp.sec = 101112;
    scan.header.frame_id = "laser_frame";
    scan.angle_min = -M_PI / 2;
    scan.angle_max = M_PI / 2;
    scan.angle_increment = M_PI / 180;  // 1 degree
    scan.range_min = 0.1f;
    scan.range_max = 10.0f;

    // Create 180 range readings
    scan.ranges.resize(180);
    for (size_t i = 0; i < scan.ranges.size(); ++i) {
      scan.ranges[i] = 1.0f + static_cast<float>(i) * 0.01f;
    }

    auto tensor = from_laserscan(scan, allocator);

    REQUIRE(tensor.shape().size() == 1);  // [num_ranges]
    REQUIRE(tensor.shape()[0] == 180);
    REQUIRE(tensor.dtype() == DataType::FLOAT32);
  }

  SECTION("LaserScan with ranges and intensities")
  {
    sensor_msgs::msg::LaserScan scan;
    scan.ranges.resize(100, 5.0f);
    scan.intensities.resize(100, 100.0f);  // Non-empty intensities

    auto tensor = from_laserscan(scan, allocator);

    REQUIRE(tensor.shape().size() == 2);  // [num_ranges, 2] for ranges + intensities
    REQUIRE(tensor.shape()[0] == 100);
    REQUIRE(tensor.shape()[1] == 2);
    REQUIRE(tensor.dtype() == DataType::FLOAT32);
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "IMU conversion", "[conversions][imu]")
{
  auto allocator = getAllocator();

  SECTION("Complete IMU data")
  {
    sensor_msgs::msg::Imu imu;
    imu.header.stamp.sec = 202425;
    imu.header.frame_id = "imu_frame";

    // Set orientation (quaternion)
    imu.orientation.x = 0.1;
    imu.orientation.y = 0.2;
    imu.orientation.z = 0.3;
    imu.orientation.w = 0.9;

    // Set linear acceleration
    imu.linear_acceleration.x = 1.0;
    imu.linear_acceleration.y = 2.0;
    imu.linear_acceleration.z = 9.8;

    // Set angular velocity
    imu.angular_velocity.x = 0.01;
    imu.angular_velocity.y = 0.02;
    imu.angular_velocity.z = 0.03;

    auto tensor = from_imu(imu, allocator);

    REQUIRE(tensor.shape().size() == 1);  // [10] for qx,qy,qz,qw,ax,ay,az,gx,gy,gz
    REQUIRE(tensor.shape()[0] == 10);
    REQUIRE(tensor.dtype() == DataType::FLOAT32);

    // Verify data order: [qx, qy, qz, qw, ax, ay, az, gx, gy, gz]
    float * data = static_cast<float *>(tensor.data());
    REQUIRE(data[0] == Approx(0.1f));  // qx
    REQUIRE(data[1] == Approx(0.2f));  // qy
    REQUIRE(data[2] == Approx(0.3f));  // qz
    REQUIRE(data[3] == Approx(0.9f));  // qw
    REQUIRE(data[4] == Approx(1.0f));  // ax
    REQUIRE(data[5] == Approx(2.0f));  // ay
    REQUIRE(data[6] == Approx(9.8f));  // az
    REQUIRE(data[7] == Approx(0.01f));  // gx
    REQUIRE(data[8] == Approx(0.02f));  // gy
    REQUIRE(data[9] == Approx(0.03f));  // gz
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "Error handling and edge cases", "[conversions][error]")
{
  auto allocator = getAllocator();

  SECTION("Invalid image encoding")
  {
    REQUIRE_THROWS(get_image_encoding_info("invalid_encoding"));
  }

  SECTION("Empty image data")
  {
    sensor_msgs::msg::Image empty_image;
    empty_image.height = 0;
    empty_image.width = 0;
    empty_image.encoding = "rgb8";

    // Should handle gracefully or throw appropriate exception
    REQUIRE_THROWS(from_image(empty_image, allocator));
  }

  SECTION("Mismatched image step size")
  {
    sensor_msgs::msg::Image bad_image;
    bad_image.height = 100;
    bad_image.width = 100;
    bad_image.encoding = "rgb8";
    bad_image.step = 100;  // Should be 300 for rgb8
    bad_image.data.resize(100 * 100 * 3);

    // Should detect step size mismatch
    REQUIRE_THROWS(from_image(bad_image, allocator));
  }

  SECTION("Empty batch conversion")
  {
    std::vector<sensor_msgs::msg::Image> empty_batch;
    REQUIRE_THROWS(from_image(empty_batch, allocator));
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "Performance and memory efficiency", "[conversions][performance]")
{
  auto allocator = getAllocator();

  SECTION("Large image processing")
  {
    sensor_msgs::msg::Image large_image;
    large_image.height = 1080;
    large_image.width = 1920;
    large_image.encoding = "rgb8";
    large_image.step = 1920 * 3;
    large_image.data.resize(1080 * 1920 * 3);

    auto start = std::chrono::high_resolution_clock::now();
    auto tensor = from_image(large_image, allocator);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    REQUIRE(tensor.data() != nullptr);
    REQUIRE(duration.count() < 100);  // Should complete within 100ms
  }

  SECTION("Memory allocation tracking")
  {
    size_t initial_bytes = allocator->allocated_bytes();

    sensor_msgs::msg::Image test_image;
    test_image.height = 100;
    test_image.width = 100;
    test_image.encoding = "rgb8";
    test_image.step = 100 * 3;
    test_image.data.resize(100 * 100 * 3);

    auto tensor = from_image(test_image, allocator);

    REQUIRE(allocator->allocated_bytes() > initial_bytes);
    REQUIRE(allocator->allocated_bytes() >= 100 * 100 * 3);  // At least image size
  }
}

}  // namespace test
}  // namespace ros_conversions
}  // namespace deep_ros
