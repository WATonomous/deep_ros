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

#include "deep_conversions/image_conversions.hpp"

#include <cstring>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

    // YUV formats - 8-bit
    {"yuv422", {DataType::UINT8, 2, 1}},
    {"YUV422_YUY2", {DataType::UINT8, 2, 1}},
    {"UYVY", {DataType::UINT8, 2, 1}},
    {"YUYV", {DataType::UINT8, 2, 1}},
    {"yuv444", {DataType::UINT8, 3, 1}},
    {"YUV444", {DataType::UINT8, 3, 1}},

    // YUV formats - 16-bit
    {"yuv422_16", {DataType::UINT16, 2, 2}},
    {"yuv444_16", {DataType::UINT16, 3, 2}},
    {"YUV422_16", {DataType::UINT16, 2, 2}},
    {"YUV444_16", {DataType::UINT16, 3, 2}},
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

Tensor from_image(
  const sensor_msgs::msg::Image & image, std::shared_ptr<BackendMemoryAllocator> allocator, TensorLayout layout)
{
  if (image.height == 0 || image.width == 0) {
    throw std::runtime_error(
      "Invalid image dimensions: height=" + std::to_string(image.height) + ", width=" + std::to_string(image.width));
  }

  auto encoding_info = get_image_encoding_info(image.encoding);

  // Create tensor with proper shape based on layout
  std::vector<size_t> shape;
  if (layout == TensorLayout::CHW) {
    // CHW: [batch, channels, height, width]
    shape = {1, encoding_info.channels, image.height, image.width};
  } else {
    // HWC: [batch, height, width, channels]
    shape = {1, image.height, image.width};
    if (encoding_info.channels > 1) {
      shape.push_back(encoding_info.channels);
    }
  }

  // Validate step size (bytes per row)
  size_t expected_step = image.width * encoding_info.channels * encoding_info.bytes_per_channel;
  if (image.step != expected_step) {
    throw std::runtime_error(
      "Image step mismatch. Expected " + std::to_string(expected_step) + " but got " + std::to_string(image.step));
  }

  // Validate data size
  size_t expected_size = image.height * image.width * encoding_info.channels * encoding_info.bytes_per_channel;
  if (image.data.size() != expected_size) {
    throw std::runtime_error(
      "Image data size mismatch. Expected " + std::to_string(expected_size) + " but got " +
      std::to_string(image.data.size()));
  }

  Tensor tensor(shape, encoding_info.dtype, allocator);

  if (layout == TensorLayout::HWC) {
    // Direct copy for HWC layout
    if (allocator) {
      allocator->copy_from_host(tensor.data(), image.data.data(), image.data.size());
    } else {
      std::memcpy(tensor.data(), image.data.data(), image.data.size());
    }
  } else {
    // Transpose HWC to CHW for CHW layout
    const auto * src = image.data.data();
    auto * dst = static_cast<uint8_t *>(tensor.data());
    size_t pixel_bytes = encoding_info.bytes_per_channel;

    for (size_t c = 0; c < encoding_info.channels; ++c) {
      for (size_t h = 0; h < image.height; ++h) {
        for (size_t w = 0; w < image.width; ++w) {
          size_t src_idx = ((h * image.width + w) * encoding_info.channels + c) * pixel_bytes;
          size_t dst_idx = ((c * image.height + h) * image.width + w) * pixel_bytes;
          std::memcpy(dst + dst_idx, src + src_idx, pixel_bytes);
        }
      }
    }
  }

  return tensor;
}

Tensor from_image(
  const std::vector<sensor_msgs::msg::Image> & images,
  std::shared_ptr<BackendMemoryAllocator> allocator,
  TensorLayout layout)
{
  if (images.empty()) {
    throw std::invalid_argument("Image batch is empty");
  }

  // Get encoding info from first image
  auto encoding_info = get_image_encoding_info(images[0].encoding);

  // Create batch shape based on layout
  std::vector<size_t> shape;
  if (layout == TensorLayout::CHW) {
    // CHW: [batch_size, channels, height, width]
    shape = {images.size(), encoding_info.channels, images[0].height, images[0].width};
  } else {
    // HWC: [batch_size, height, width, channels]
    shape = {images.size(), images[0].height, images[0].width};
    if (encoding_info.channels > 1) {
      shape.push_back(encoding_info.channels);
    }
  }

  // Validate all images have same dimensions and encoding
  size_t expected_size = images[0].height * images[0].width * encoding_info.channels * encoding_info.bytes_per_channel;
  for (size_t i = 0; i < images.size(); ++i) {
    if (images[i].height != images[0].height || images[i].width != images[0].width) {
      throw std::invalid_argument("All images in batch must have same dimensions");
    }
    if (images[i].encoding != images[0].encoding) {
      throw std::invalid_argument("All images in batch must have same encoding");
    }
    if (images[i].data.size() != expected_size) {
      throw std::runtime_error(
        "Image " + std::to_string(i) + " data size mismatch. Expected " + std::to_string(expected_size) + " but got " +
        std::to_string(images[i].data.size()));
    }
  }

  Tensor tensor(shape, encoding_info.dtype, allocator);
  auto * dst = static_cast<uint8_t *>(tensor.data());
  size_t height = images[0].height;
  size_t width = images[0].width;
  size_t pixel_bytes = encoding_info.bytes_per_channel;

  if (layout == TensorLayout::HWC) {
    // Direct copy for HWC layout
    for (size_t i = 0; i < images.size(); ++i) {
      if (allocator) {
        allocator->copy_from_host(dst + i * images[i].data.size(), images[i].data.data(), images[i].data.size());
      } else {
        std::memcpy(dst + i * images[i].data.size(), images[i].data.data(), images[i].data.size());
      }
    }
  } else {
    // Transpose HWC to CHW for each image in batch
    for (size_t b = 0; b < images.size(); ++b) {
      const auto * src = images[b].data.data();
      size_t batch_offset = b * encoding_info.channels * height * width * pixel_bytes;

      for (size_t c = 0; c < encoding_info.channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            size_t src_idx = ((h * width + w) * encoding_info.channels + c) * pixel_bytes;
            size_t dst_idx = batch_offset + ((c * height + h) * width + w) * pixel_bytes;
            std::memcpy(dst + dst_idx, src + src_idx, pixel_bytes);
          }
        }
      }
    }
  }

  return tensor;
}

void to_image(
  const Tensor & tensor,
  sensor_msgs::msg::Image & image,
  const std::string & encoding,
  const std_msgs::msg::Header & header)
{
  if (tensor.rank() < 3 || tensor.rank() > 4) {
    throw std::invalid_argument(
      "Tensor must be 3D [batch, height, width] or 4D [batch, height, width, channels] for image conversion");
  }

  if (tensor.shape()[0] != 1) {
    throw std::invalid_argument(
      "This overload only supports single images (batch size must be 1). Use vector overload for multiple images.");
  }

  image.header = header;

  // Verify encoding matches tensor
  auto encoding_info = get_image_encoding_info(encoding);
  size_t tensor_channels = (tensor.rank() == 4) ? tensor.shape()[3] : 1;

  if (encoding_info.channels != tensor_channels) {
    throw std::invalid_argument(
      "Encoding channels (" + std::to_string(encoding_info.channels) + ") doesn't match tensor channels (" +
      std::to_string(tensor_channels) + ")");
  }

  if (encoding_info.dtype != tensor.dtype()) {
    throw std::invalid_argument("Encoding data type doesn't match tensor data type");
  }

  image.height = tensor.shape()[1];  // Skip batch dimension
  image.width = tensor.shape()[2];  // Skip batch dimension
  image.encoding = encoding;

  // Calculate step (bytes per row)
  image.step = image.width * encoding_info.channels * encoding_info.bytes_per_channel;

  // Set big endian flag (false for little endian)
  image.is_bigendian = false;

  // Copy tensor data to image (skip batch dimension)
  size_t image_size = image.height * image.width * encoding_info.channels * encoding_info.bytes_per_channel;
  image.data.resize(image_size);
  std::memcpy(image.data.data(), tensor.data(), image_size);
}

void to_image(
  const Tensor & tensor,
  std::vector<sensor_msgs::msg::Image> & images,
  const std::string & encoding,
  const std_msgs::msg::Header & header)
{
  if (tensor.rank() < 3 || tensor.rank() > 4) {
    throw std::invalid_argument(
      "Tensor must be 3D [batch, height, width] or 4D [batch, height, width, channels] for image conversion");
  }

  size_t batch_size = tensor.shape()[0];
  if (batch_size == 0) {
    throw std::invalid_argument("Batch size cannot be 0");
  }

  // Verify encoding matches tensor
  auto encoding_info = get_image_encoding_info(encoding);
  size_t tensor_channels = (tensor.rank() == 4) ? tensor.shape()[3] : 1;

  if (encoding_info.channels != tensor_channels) {
    throw std::invalid_argument(
      "Encoding channels (" + std::to_string(encoding_info.channels) + ") doesn't match tensor channels (" +
      std::to_string(tensor_channels) + ")");
  }

  if (encoding_info.dtype != tensor.dtype()) {
    throw std::invalid_argument("Encoding data type doesn't match tensor data type");
  }

  size_t height = tensor.shape()[1];
  size_t width = tensor.shape()[2];
  size_t image_size = height * width * encoding_info.channels * encoding_info.bytes_per_channel;

  images.clear();
  images.reserve(batch_size);

  const auto * src = static_cast<const uint8_t *>(tensor.data());

  for (size_t i = 0; i < batch_size; ++i) {
    sensor_msgs::msg::Image image;
    image.header = header;
    image.height = height;
    image.width = width;
    image.encoding = encoding;
    image.step = width * encoding_info.channels * encoding_info.bytes_per_channel;
    image.is_bigendian = false;

    image.data.resize(image_size);
    std::memcpy(image.data.data(), src + i * image_size, image_size);

    images.push_back(std::move(image));
  }
}

}  // namespace ros_conversions
}  // namespace deep_ros
