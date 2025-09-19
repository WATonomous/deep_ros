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

#include "deep_tensor/ros_conversions.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace deep_ros
{
namespace ros_conversions
{

ImageEncoding get_image_encoding_info(const std::string & encoding)
{
  static const std::unordered_map<std::string, ImageEncoding> encoding_map = {
    // RGB/BGR formats
    {"rgb8", {DataType::UINT8, 3, 1}},
    {"rgba8", {DataType::UINT8, 4, 1}},
    {"rgb16", {DataType::UINT16, 3, 2}},
    {"rgba16", {DataType::UINT16, 4, 2}},
    {"bgr8", {DataType::UINT8, 3, 1}},
    {"bgra8", {DataType::UINT8, 4, 1}},
    {"bgr16", {DataType::UINT16, 3, 2}},
    {"bgra16", {DataType::UINT16, 4, 2}},

    // Grayscale
    {"mono8", {DataType::UINT8, 1, 1}},
    {"mono16", {DataType::UINT16, 1, 2}},

    // Bayer patterns (treat as single channel)
    {"bayer_rggb8", {DataType::UINT8, 1, 1}},
    {"bayer_bggr8", {DataType::UINT8, 1, 1}},
    {"bayer_gbrg8", {DataType::UINT8, 1, 1}},
    {"bayer_grbg8", {DataType::UINT8, 1, 1}},
    {"bayer_rggb16", {DataType::UINT16, 1, 2}},
    {"bayer_bggr16", {DataType::UINT16, 1, 2}},
    {"bayer_gbrg16", {DataType::UINT16, 1, 2}},
    {"bayer_grbg16", {DataType::UINT16, 1, 2}},

    // OpenCV type formats
    {"8UC1", {DataType::UINT8, 1, 1}},
    {"8UC2", {DataType::UINT8, 2, 1}},
    {"8UC3", {DataType::UINT8, 3, 1}},
    {"8UC4", {DataType::UINT8, 4, 1}},
    {"8SC1", {DataType::INT8, 1, 1}},
    {"8SC2", {DataType::INT8, 2, 1}},
    {"8SC3", {DataType::INT8, 3, 1}},
    {"8SC4", {DataType::INT8, 4, 1}},
    {"16UC1", {DataType::UINT16, 1, 2}},
    {"16UC2", {DataType::UINT16, 2, 2}},
    {"16UC3", {DataType::UINT16, 3, 2}},
    {"16UC4", {DataType::UINT16, 4, 2}},
    {"16SC1", {DataType::INT16, 1, 2}},
    {"16SC2", {DataType::INT16, 2, 2}},
    {"16SC3", {DataType::INT16, 3, 2}},
    {"16SC4", {DataType::INT16, 4, 2}},
    {"32SC1", {DataType::INT32, 1, 4}},
    {"32SC2", {DataType::INT32, 2, 4}},
    {"32SC3", {DataType::INT32, 3, 4}},
    {"32SC4", {DataType::INT32, 4, 4}},
    {"32FC1", {DataType::FLOAT32, 1, 4}},
    {"32FC2", {DataType::FLOAT32, 2, 4}},
    {"32FC3", {DataType::FLOAT32, 3, 4}},
    {"32FC4", {DataType::FLOAT32, 4, 4}},
    {"64FC1", {DataType::FLOAT64, 1, 8}},
    {"64FC2", {DataType::FLOAT64, 2, 8}},
    {"64FC3", {DataType::FLOAT64, 3, 8}},
    {"64FC4", {DataType::FLOAT64, 4, 8}},

    // YUV formats (treat as 2 channel for YUV422)
    {"yuv422", {DataType::UINT8, 2, 1}},
    {"YUV422_YUY2", {DataType::UINT8, 2, 1}},
    {"UYVY", {DataType::UINT8, 2, 1}},
    {"YUYV", {DataType::UINT8, 2, 1}},
  };

  auto it = encoding_map.find(encoding);
  if (it != encoding_map.end()) {
    return it->second;
  }

  // Try to parse TYPE_XX(C)N format using regex
  std::regex type_regex("(8U|8S|16U|16S|32S|32F|64F)C([1-4])");
  std::smatch match;
  if (std::regex_match(encoding, match, type_regex)) {
    std::string type_str = match[1];
    int channels = std::stoi(match[2]);

    if (type_str == "8U") return {DataType::UINT8, static_cast<size_t>(channels), 1};
    if (type_str == "8S") return {DataType::INT8, static_cast<size_t>(channels), 1};
    if (type_str == "16U") return {DataType::UINT16, static_cast<size_t>(channels), 2};
    if (type_str == "16S") return {DataType::INT16, static_cast<size_t>(channels), 2};
    if (type_str == "32S") return {DataType::INT32, static_cast<size_t>(channels), 4};
    if (type_str == "32F") return {DataType::FLOAT32, static_cast<size_t>(channels), 4};
    if (type_str == "64F") return {DataType::FLOAT64, static_cast<size_t>(channels), 8};
  }

  throw std::runtime_error("Unsupported image encoding: " + encoding);
}

Tensor from_image(const sensor_msgs::msg::Image & image, bool normalize)
{
  auto encoding_info = get_image_encoding_info(image.encoding);

  // Create tensor with proper shape
  std::vector<size_t> shape = {image.height, image.width};
  if (encoding_info.channels > 1) {
    shape.push_back(encoding_info.channels);
  }

  // Validate data size
  size_t expected_size = image.height * image.width * encoding_info.channels * encoding_info.bytes_per_channel;
  if (image.data.size() != expected_size) {
    throw std::runtime_error(
      "Image data size mismatch. Expected " + std::to_string(expected_size) + " but got " +
      std::to_string(image.data.size()));
  }

  if (normalize && encoding_info.dtype == DataType::UINT8) {
    // Convert to float32 and normalize
    Tensor tensor(shape, DataType::FLOAT32);
    auto * dst = tensor.data_as<float>();
    const auto * src = image.data.data();

    for (size_t i = 0; i < image.data.size(); ++i) {
      dst[i] = static_cast<float>(src[i]) / 255.0f;
    }
    return tensor;
  } else {
    // Direct copy
    Tensor tensor(shape, encoding_info.dtype);
    std::memcpy(tensor.data(), image.data.data(), image.data.size());
    return tensor;
  }
}

sensor_msgs::msg::Image to_image(
  const Tensor & tensor, const std::string & encoding, const std_msgs::msg::Header & header)
{
  sensor_msgs::msg::Image image;
  image.header = header;

  if (tensor.rank() < 2 || tensor.rank() > 3) {
    throw std::invalid_argument("Tensor must be 2D or 3D for image conversion");
  }

  // Verify encoding matches tensor
  auto encoding_info = get_image_encoding_info(encoding);
  size_t tensor_channels = (tensor.rank() == 3) ? tensor.shape()[2] : 1;

  if (encoding_info.channels != tensor_channels) {
    throw std::invalid_argument(
      "Encoding channels (" + std::to_string(encoding_info.channels) + ") doesn't match tensor channels (" +
      std::to_string(tensor_channels) + ")");
  }

  if (encoding_info.dtype != tensor.dtype()) {
    throw std::invalid_argument("Encoding data type doesn't match tensor data type");
  }

  image.height = tensor.shape()[0];
  image.width = tensor.shape()[1];
  image.encoding = encoding;

  // Calculate step (bytes per row)
  image.step = image.width * encoding_info.channels * encoding_info.bytes_per_channel;

  // Set big endian flag (false for little endian)
  image.is_bigendian = false;

  // Copy tensor data to image
  size_t data_size = tensor.byte_size();
  image.data.resize(data_size);
  std::memcpy(image.data.data(), tensor.data(), data_size);

  return image;
}

Tensor from_pointcloud2(const sensor_msgs::msg::PointCloud2 & cloud)
{
  if (cloud.fields.empty()) {
    throw std::invalid_argument("PointCloud2 has no fields");
  }

  size_t num_points = cloud.width * cloud.height;
  size_t num_fields = cloud.fields.size();

  // For simplicity, convert all to float32
  std::vector<size_t> shape = {num_points, num_fields};
  Tensor tensor(shape, DataType::FLOAT32);
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

Tensor from_laserscan(const sensor_msgs::msg::LaserScan & scan)
{
  size_t num_ranges = scan.ranges.size();
  bool has_intensities = !scan.intensities.empty() && scan.intensities.size() == num_ranges;

  if (has_intensities) {
    std::vector<size_t> shape = {num_ranges, 2};
    Tensor tensor(shape, DataType::FLOAT32);
    auto * data = tensor.data_as<float>();

    for (size_t i = 0; i < num_ranges; ++i) {
      data[i * 2] = scan.ranges[i];
      data[i * 2 + 1] = scan.intensities[i];
    }
    return tensor;
  } else {
    std::vector<size_t> shape = {num_ranges};
    Tensor tensor(shape, DataType::FLOAT32);
    std::memcpy(tensor.data(), scan.ranges.data(), num_ranges * sizeof(float));
    return tensor;
  }
}

Tensor from_imu(const sensor_msgs::msg::Imu & imu)
{
  std::vector<size_t> shape = {10};
  Tensor tensor(shape, DataType::FLOAT64);
  auto * data = tensor.data_as<double>();

  // Orientation quaternion
  data[0] = imu.orientation.x;
  data[1] = imu.orientation.y;
  data[2] = imu.orientation.z;
  data[3] = imu.orientation.w;

  // Linear acceleration
  data[4] = imu.linear_acceleration.x;
  data[5] = imu.linear_acceleration.y;
  data[6] = imu.linear_acceleration.z;

  // Angular velocity
  data[7] = imu.angular_velocity.x;
  data[8] = imu.angular_velocity.y;
  data[9] = imu.angular_velocity.z;

  return tensor;
}

Tensor from_image_batch(const std::vector<sensor_msgs::msg::Image> & images, bool normalize)
{
  if (images.empty()) {
    throw std::invalid_argument("Image batch is empty");
  }

  // Get dimensions from first image
  auto first_tensor = from_image(images[0], normalize);
  auto shape = first_tensor.shape();

  // Create batch shape
  std::vector<size_t> batch_shape = {images.size()};
  batch_shape.insert(batch_shape.end(), shape.begin(), shape.end());

  Tensor batch_tensor(batch_shape, first_tensor.dtype());

  // Copy each image
  size_t image_size = first_tensor.byte_size();
  uint8_t * dst = static_cast<uint8_t *>(batch_tensor.data());

  std::memcpy(dst, first_tensor.data(), image_size);

  for (size_t i = 1; i < images.size(); ++i) {
    auto img_tensor = from_image(images[i], normalize);
    if (img_tensor.shape() != shape) {
      throw std::invalid_argument("All images in batch must have same dimensions");
    }

    dst += image_size;
    std::memcpy(dst, img_tensor.data(), image_size);
  }

  return batch_tensor;
}

}  // namespace ros_conversions
}  // namespace deep_ros
