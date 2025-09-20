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

#include <vector>

#include <catch2/catch.hpp>

#include "deep_conversions/ros_conversions.hpp"

TEST_CASE("Image encoding info parsing", "[ros_conversions]")
{
  SECTION("RGB encodings")
  {
    auto rgb8_info = deep_ros::ros_conversions::get_image_encoding_info("rgb8");
    REQUIRE(rgb8_info.dtype == deep_ros::DataType::UINT8);
    REQUIRE(rgb8_info.channels == 3);
    REQUIRE(rgb8_info.bytes_per_channel == 1);

    auto rgba16_info = deep_ros::ros_conversions::get_image_encoding_info("rgba16");
    REQUIRE(rgba16_info.dtype == deep_ros::DataType::UINT16);
    REQUIRE(rgba16_info.channels == 4);
    REQUIRE(rgba16_info.bytes_per_channel == 2);
  }

  SECTION("Grayscale encodings")
  {
    auto mono8_info = deep_ros::ros_conversions::get_image_encoding_info("mono8");
    REQUIRE(mono8_info.dtype == deep_ros::DataType::UINT8);
    REQUIRE(mono8_info.channels == 1);
    REQUIRE(mono8_info.bytes_per_channel == 1);

    auto mono16_info = deep_ros::ros_conversions::get_image_encoding_info("mono16");
    REQUIRE(mono16_info.dtype == deep_ros::DataType::UINT16);
    REQUIRE(mono16_info.channels == 1);
    REQUIRE(mono16_info.bytes_per_channel == 2);
  }

  SECTION("OpenCV type encodings")
  {
    auto type_info = deep_ros::ros_conversions::get_image_encoding_info("32FC3");
    REQUIRE(type_info.dtype == deep_ros::DataType::FLOAT32);
    REQUIRE(type_info.channels == 3);
    REQUIRE(type_info.bytes_per_channel == 4);

    auto signed_info = deep_ros::ros_conversions::get_image_encoding_info("16SC2");
    REQUIRE(signed_info.dtype == deep_ros::DataType::INT16);
    REQUIRE(signed_info.channels == 2);
    REQUIRE(signed_info.bytes_per_channel == 2);
  }

  SECTION("Bayer patterns")
  {
    auto bayer_info = deep_ros::ros_conversions::get_image_encoding_info("bayer_rggb8");
    REQUIRE(bayer_info.dtype == deep_ros::DataType::UINT8);
    REQUIRE(bayer_info.channels == 1);
    REQUIRE(bayer_info.bytes_per_channel == 1);
  }

  SECTION("YUV formats - 8-bit")
  {
    auto yuv422_info = deep_ros::ros_conversions::get_image_encoding_info("yuv422");
    REQUIRE(yuv422_info.dtype == deep_ros::DataType::UINT8);
    REQUIRE(yuv422_info.channels == 2);
    REQUIRE(yuv422_info.bytes_per_channel == 1);

    auto yuv444_info = deep_ros::ros_conversions::get_image_encoding_info("yuv444");
    REQUIRE(yuv444_info.dtype == deep_ros::DataType::UINT8);
    REQUIRE(yuv444_info.channels == 3);
    REQUIRE(yuv444_info.bytes_per_channel == 1);
  }

  SECTION("YUV formats - 16-bit")
  {
    auto yuv422_16_info = deep_ros::ros_conversions::get_image_encoding_info("yuv422_16");
    REQUIRE(yuv422_16_info.dtype == deep_ros::DataType::UINT16);
    REQUIRE(yuv422_16_info.channels == 2);
    REQUIRE(yuv422_16_info.bytes_per_channel == 2);

    auto yuv444_16_info = deep_ros::ros_conversions::get_image_encoding_info("yuv444_16");
    REQUIRE(yuv444_16_info.dtype == deep_ros::DataType::UINT16);
    REQUIRE(yuv444_16_info.channels == 3);
    REQUIRE(yuv444_16_info.bytes_per_channel == 2);
  }

  SECTION("Unsupported encoding")
  {
    REQUIRE_THROWS_AS(deep_ros::ros_conversions::get_image_encoding_info("invalid_encoding"), std::runtime_error);
  }
}

TEST_CASE("Image to tensor conversion", "[ros_conversions]")
{
  SECTION("RGB8 image")
  {
    sensor_msgs::msg::Image image;
    image.width = 2;
    image.height = 2;
    image.encoding = "rgb8";
    image.step = 6;  // 2 pixels * 3 channels * 1 byte
    image.data = {
      255,
      0,
      0,  // Red pixel
      0,
      255,
      0,  // Green pixel
      0,
      0,
      255,  // Blue pixel
      128,
      128,
      128};  // Gray pixel

    auto tensor = deep_ros::ros_conversions::from_image(image);

    REQUIRE(tensor.shape() == std::vector<size_t>{1, 2, 2, 3});
    REQUIRE(tensor.dtype() == deep_ros::DataType::UINT8);
    REQUIRE(tensor.size() == 12);

    auto data = tensor.data_as<uint8_t>();
    REQUIRE(data[0] == 255);  // First pixel R
    REQUIRE(data[1] == 0);  // First pixel G
    REQUIRE(data[2] == 0);  // First pixel B
  }

  SECTION("Mono8 image")
  {
    sensor_msgs::msg::Image image;
    image.width = 3;
    image.height = 2;
    image.encoding = "mono8";
    image.step = 3;
    image.data = {100, 150, 200, 50, 75, 25};

    auto tensor = deep_ros::ros_conversions::from_image(image);

    REQUIRE(tensor.shape() == std::vector<size_t>{1, 2, 3});
    REQUIRE(tensor.dtype() == deep_ros::DataType::UINT8);
    REQUIRE(tensor.size() == 6);

    auto data = tensor.data_as<uint8_t>();
    REQUIRE(data[0] == 100);
    REQUIRE(data[3] == 50);
  }

  SECTION("32FC1 image")
  {
    sensor_msgs::msg::Image image;
    image.width = 2;
    image.height = 1;
    image.encoding = "32FC1";
    image.step = 8;  // 2 pixels * 1 channel * 4 bytes

    std::vector<float> float_data = {1.5f, -2.5f};
    image.data.resize(8);
    std::memcpy(image.data.data(), float_data.data(), 8);

    auto tensor = deep_ros::ros_conversions::from_image(image);

    REQUIRE(tensor.shape() == std::vector<size_t>{1, 1, 2});
    REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);

    auto data = tensor.data_as<float>();
    REQUIRE(data[0] == 1.5f);
    REQUIRE(data[1] == -2.5f);
  }

  SECTION("Batch of single image")
  {
    sensor_msgs::msg::Image image;
    image.width = 2;
    image.height = 1;
    image.encoding = "mono8";
    image.step = 2;
    image.data = {100, 200};

    auto tensor = deep_ros::ros_conversions::from_image(image);

    REQUIRE(tensor.shape() == std::vector<size_t>{1, 1, 2});
    REQUIRE(tensor.dtype() == deep_ros::DataType::UINT8);
    auto data = tensor.data_as<uint8_t>();
    REQUIRE(data[0] == 100);
    REQUIRE(data[1] == 200);
  }
}

TEST_CASE("Tensor to image conversion", "[ros_conversions]")
{
  SECTION("RGB tensor to image")
  {
    std::vector<size_t> shape = {1, 2, 2, 3};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::UINT8);

    auto data = tensor.data_as<uint8_t>();
    data[0] = 255;
    data[1] = 0;
    data[2] = 0;  // Red
    data[3] = 0;
    data[4] = 255;
    data[5] = 0;  // Green
    data[6] = 0;
    data[7] = 0;
    data[8] = 255;  // Blue
    data[9] = 128;
    data[10] = 128;
    data[11] = 128;  // Gray

    sensor_msgs::msg::Image image;
    deep_ros::ros_conversions::to_image(tensor, image, "rgb8");

    REQUIRE(image.width == 2);
    REQUIRE(image.height == 2);
    REQUIRE(image.encoding == "rgb8");
    REQUIRE(image.step == 6);
    REQUIRE(image.data.size() == 12);
    REQUIRE(image.data[0] == 255);
    REQUIRE(image.data[9] == 128);
  }

  SECTION("Grayscale tensor to image")
  {
    std::vector<size_t> shape = {1, 1, 3};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::UINT8);

    auto data = tensor.data_as<uint8_t>();
    data[0] = 100;
    data[1] = 150;
    data[2] = 200;

    sensor_msgs::msg::Image image;
    deep_ros::ros_conversions::to_image(tensor, image, "mono8");

    REQUIRE(image.width == 3);
    REQUIRE(image.height == 1);
    REQUIRE(image.encoding == "mono8");
    REQUIRE(image.step == 3);
    REQUIRE(image.data[1] == 150);
  }

  SECTION("Type mismatch error")
  {
    std::vector<size_t> shape = {1, 2, 2};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32);

    sensor_msgs::msg::Image image;
    REQUIRE_THROWS_AS(deep_ros::ros_conversions::to_image(tensor, image, "mono8"), std::invalid_argument);
  }

  SECTION("Channel mismatch error")
  {
    std::vector<size_t> shape = {1, 2, 2, 3};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::UINT8);

    sensor_msgs::msg::Image image;
    REQUIRE_THROWS_AS(deep_ros::ros_conversions::to_image(tensor, image, "mono8"), std::invalid_argument);
  }
}

TEST_CASE("LaserScan conversion", "[ros_conversions]")
{
  SECTION("Ranges only")
  {
    sensor_msgs::msg::LaserScan scan;
    scan.ranges = {1.0f, 2.0f, 3.0f, 4.0f};

    auto tensor = deep_ros::ros_conversions::from_laserscan(scan);

    REQUIRE(tensor.shape() == std::vector<size_t>{4});
    REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);

    auto data = tensor.data_as<float>();
    REQUIRE(data[0] == 1.0f);
    REQUIRE(data[3] == 4.0f);
  }

  SECTION("Ranges and intensities")
  {
    sensor_msgs::msg::LaserScan scan;
    scan.ranges = {1.0f, 2.0f};
    scan.intensities = {0.8f, 0.9f};

    auto tensor = deep_ros::ros_conversions::from_laserscan(scan);

    REQUIRE(tensor.shape() == std::vector<size_t>{2, 2});
    REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);

    auto data = tensor.data_as<float>();
    REQUIRE(data[0] == 1.0f);  // First range
    REQUIRE(data[1] == 0.8f);  // First intensity
    REQUIRE(data[2] == 2.0f);  // Second range
    REQUIRE(data[3] == 0.9f);  // Second intensity
  }
}

TEST_CASE("IMU conversion", "[ros_conversions]")
{
  sensor_msgs::msg::Imu imu;

  // Set orientation quaternion
  imu.orientation.x = 0.1;
  imu.orientation.y = 0.2;
  imu.orientation.z = 0.3;
  imu.orientation.w = 0.9;

  // Set linear acceleration
  imu.linear_acceleration.x = 1.0;
  imu.linear_acceleration.y = 2.0;
  imu.linear_acceleration.z = 9.8;

  // Set angular velocity
  imu.angular_velocity.x = 0.1;
  imu.angular_velocity.y = 0.2;
  imu.angular_velocity.z = 0.3;

  auto tensor = deep_ros::ros_conversions::from_imu(imu);

  REQUIRE(tensor.shape() == std::vector<size_t>{10});
  REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT64);

  auto data = tensor.data_as<double>();

  // Check orientation quaternion
  REQUIRE(data[0] == 0.1);  // qx
  REQUIRE(data[1] == 0.2);  // qy
  REQUIRE(data[2] == 0.3);  // qz
  REQUIRE(data[3] == 0.9);  // qw

  // Check linear acceleration
  REQUIRE(data[4] == 1.0);  // ax
  REQUIRE(data[5] == 2.0);  // ay
  REQUIRE(data[6] == 9.8);  // az

  // Check angular velocity
  REQUIRE(data[7] == 0.1);  // gx
  REQUIRE(data[8] == 0.2);  // gy
  REQUIRE(data[9] == 0.3);  // gz
}

TEST_CASE("Image batch conversion", "[ros_conversions]")
{
  SECTION("Valid batch")
  {
    std::vector<sensor_msgs::msg::Image> images(2);

    // First image
    images[0].width = 2;
    images[0].height = 1;
    images[0].encoding = "mono8";
    images[0].step = 2;
    images[0].data = {100, 150};

    // Second image
    images[1].width = 2;
    images[1].height = 1;
    images[1].encoding = "mono8";
    images[1].step = 2;
    images[1].data = {200, 250};

    auto tensor = deep_ros::ros_conversions::from_image(images);

    REQUIRE(tensor.shape() == std::vector<size_t>{2, 1, 2});
    REQUIRE(tensor.dtype() == deep_ros::DataType::UINT8);

    auto data = tensor.data_as<uint8_t>();
    REQUIRE(data[0] == 100);  // First image, first pixel
    REQUIRE(data[1] == 150);  // First image, second pixel
    REQUIRE(data[2] == 200);  // Second image, first pixel
    REQUIRE(data[3] == 250);  // Second image, second pixel
  }

  SECTION("Size mismatch error")
  {
    std::vector<sensor_msgs::msg::Image> images(2);

    images[0].width = 2;
    images[0].height = 1;
    images[0].encoding = "mono8";
    images[0].data = {100, 150};

    images[1].width = 3;  // Different size
    images[1].height = 1;
    images[1].encoding = "mono8";
    images[1].data = {200, 250, 255};

    REQUIRE_THROWS_AS(deep_ros::ros_conversions::from_image(images), std::invalid_argument);
  }
}

TEST_CASE("PointCloud2 conversion", "[ros_conversions]")
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.width = 2;
  cloud.height = 1;
  cloud.point_step = 12;  // 3 fields * 4 bytes each

  // Define fields (x, y, z)
  cloud.fields.resize(3);
  cloud.fields[0].name = "x";
  cloud.fields[0].offset = 0;
  cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[0].count = 1;

  cloud.fields[1].name = "y";
  cloud.fields[1].offset = 4;
  cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[1].count = 1;

  cloud.fields[2].name = "z";
  cloud.fields[2].offset = 8;
  cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[2].count = 1;

  // Add point data: (1.0, 2.0, 3.0) and (4.0, 5.0, 6.0)
  cloud.data.resize(24);  // 2 points * 12 bytes each
  std::vector<float> point_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::memcpy(cloud.data.data(), point_data.data(), 24);

  auto tensor = deep_ros::ros_conversions::from_pointcloud2(cloud);

  REQUIRE(tensor.shape() == std::vector<size_t>{2, 3});
  REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);

  auto data = tensor.data_as<float>();
  REQUIRE(data[0] == 1.0f);  // First point x
  REQUIRE(data[1] == 2.0f);  // First point y
  REQUIRE(data[2] == 3.0f);  // First point z
  REQUIRE(data[3] == 4.0f);  // Second point x
  REQUIRE(data[4] == 5.0f);  // Second point y
  REQUIRE(data[5] == 6.0f);  // Second point z
}
