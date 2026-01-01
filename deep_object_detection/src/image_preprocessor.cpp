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

#include "deep_object_detection/image_preprocessor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace deep_object_detection
{

ImagePreprocessor::ImagePreprocessor(const PreprocessingConfig & config)
: config_(config)
{}

cv::Mat ImagePreprocessor::preprocess(const cv::Mat & bgr, ImageMeta & meta) const
{
  if (bgr.empty()) {
    throw std::runtime_error("Input image is empty");
  }

  meta.original_width = bgr.cols;
  meta.original_height = bgr.rows;

  cv::Mat resized;
  
  switch (config_.resize_method) {
    case ResizeMethod::LETTERBOX:
      resized = applyLetterbox(bgr, meta);
      break;
    case ResizeMethod::CROP:
      resized = applyCrop(bgr, meta);
      break;
    case ResizeMethod::PAD:
      resized = applyPad(bgr, meta);
      break;
    case ResizeMethod::RESIZE:
    default:
      resized = applyResize(bgr, meta);
      break;
  }

  // Convert to float and normalize
  cv::Mat float_image;
  resized.convertTo(float_image, CV_32F, 1.0 / 255.0);
  
  // Apply normalization
  applyNormalization(float_image);
  
  // Apply color conversion if needed
  applyColorConversion(float_image);
  
  return float_image;
}

cv::Mat ImagePreprocessor::applyLetterbox(const cv::Mat & bgr, ImageMeta & meta) const
{
  const float scale = std::min(
    static_cast<float>(config_.input_width) / static_cast<float>(bgr.cols),
    static_cast<float>(config_.input_height) / static_cast<float>(bgr.rows));
  const int new_w = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
  const int new_h = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));
  
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(new_w, new_h));

  const int pad_w = config_.input_width - new_w;
  const int pad_h = config_.input_height - new_h;
  const int pad_left = pad_w / 2;
  const int pad_right = pad_w - pad_left;
  const int pad_top = pad_h / 2;
  const int pad_bottom = pad_h - pad_top;

  cv::copyMakeBorder(
    resized, resized, pad_top, pad_bottom, pad_left, pad_right, 
    cv::BORDER_CONSTANT, cv::Scalar(config_.pad_value, config_.pad_value, config_.pad_value));

  meta.scale_x = meta.scale_y = scale;
  meta.pad_x = static_cast<float>(pad_left);
  meta.pad_y = static_cast<float>(pad_top);
  
  return resized;
}

cv::Mat ImagePreprocessor::applyResize(const cv::Mat & bgr, ImageMeta & meta) const
{
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(config_.input_width, config_.input_height));
  meta.scale_x = static_cast<float>(meta.original_width) / static_cast<float>(config_.input_width);
  meta.scale_y = static_cast<float>(meta.original_height) / static_cast<float>(config_.input_height);
  meta.pad_x = meta.pad_y = 0.0f;
  return resized;
}

cv::Mat ImagePreprocessor::applyCrop(const cv::Mat & bgr, ImageMeta & meta) const
{
  // Center crop
  const int crop_x = (bgr.cols - config_.input_width) / 2;
  const int crop_y = (bgr.rows - config_.input_height) / 2;
  
  cv::Mat cropped;
  if (bgr.cols >= config_.input_width && bgr.rows >= config_.input_height) {
    cropped = bgr(cv::Rect(crop_x, crop_y, config_.input_width, config_.input_height)).clone();
  } else {
    // If image is smaller than target, resize first then crop
    float scale = std::max(
      static_cast<float>(config_.input_width) / bgr.cols,
      static_cast<float>(config_.input_height) / bgr.rows);
    cv::Mat scaled;
    cv::resize(bgr, scaled, cv::Size(), scale, scale);
    const int new_crop_x = (scaled.cols - config_.input_width) / 2;
    const int new_crop_y = (scaled.rows - config_.input_height) / 2;
    cropped = scaled(cv::Rect(new_crop_x, new_crop_y, config_.input_width, config_.input_height)).clone();
  }
  
  meta.scale_x = static_cast<float>(meta.original_width) / static_cast<float>(config_.input_width);
  meta.scale_y = static_cast<float>(meta.original_height) / static_cast<float>(config_.input_height);
  meta.pad_x = static_cast<float>(-crop_x);
  meta.pad_y = static_cast<float>(-crop_y);
  
  return cropped;
}

cv::Mat ImagePreprocessor::applyPad(const cv::Mat & bgr, ImageMeta & meta) const
{
  // Pad image to target size without resizing
  const int pad_w = config_.input_width - bgr.cols;
  const int pad_h = config_.input_height - bgr.rows;
  
  cv::Mat padded;
  if (pad_w >= 0 && pad_h >= 0) {
    const int pad_left = pad_w / 2;
    const int pad_right = pad_w - pad_left;
    const int pad_top = pad_h / 2;
    const int pad_bottom = pad_h - pad_top;
    
    cv::copyMakeBorder(
      bgr, padded, pad_top, pad_bottom, pad_left, pad_right,
      cv::BORDER_CONSTANT, cv::Scalar(config_.pad_value, config_.pad_value, config_.pad_value));
    
    meta.pad_x = static_cast<float>(pad_left);
    meta.pad_y = static_cast<float>(pad_top);
  } else {
    // Image larger than target, resize first
    return applyResize(bgr, meta);
  }
  
  meta.scale_x = meta.scale_y = 1.0f;
  return padded;
}

void ImagePreprocessor::applyNormalization(cv::Mat & image) const
{
  switch (config_.normalization_type) {
    case NormalizationType::IMAGENET: {
      // ImageNet normalization: (pixel - mean) / std
      // Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225] (RGB order)
      static const std::array<float, 3> imagenet_mean = {0.485f, 0.456f, 0.406f};
      static const std::array<float, 3> imagenet_std = {0.229f, 0.224f, 0.225f};
      
      std::vector<cv::Mat> channels;
      cv::split(image, channels);
      
      // Note: OpenCV loads as BGR, so we apply in reverse order
      channels[0] = (channels[0] - imagenet_mean[2]) / imagenet_std[2];  // B
      channels[1] = (channels[1] - imagenet_mean[1]) / imagenet_std[1];  // G
      channels[2] = (channels[2] - imagenet_mean[0]) / imagenet_std[0];  // R
      
      cv::merge(channels, image);
      break;
    }
    case NormalizationType::CUSTOM: {
      if (config_.mean.size() >= 3 && config_.std.size() >= 3) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        
        // Apply custom mean/std (assuming BGR order in config)
        for (size_t i = 0; i < 3; ++i) {
          channels[i] = (channels[i] - config_.mean[i]) / config_.std[i];
        }
        
        cv::merge(channels, image);
      }
      break;
    }
    case NormalizationType::SCALE_0_1:
    case NormalizationType::NONE:
    default:
      // Scale normalization is just 0-255 to 0-1, already done in convertTo
      // NONE means no additional normalization
      break;
  }
}

void ImagePreprocessor::applyColorConversion(cv::Mat & image) const
{
  if (config_.color_format == "rgb" || config_.color_format == "RGB") {
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  }
  // If "bgr" or anything else, keep as BGR (default from OpenCV)
}

const PackedInput & ImagePreprocessor::pack(const std::vector<cv::Mat> & images) const
{
  auto & packed = packed_input_cache_;
  packed.data.clear();
  packed.shape.clear();
  if (images.empty()) {
    return packed;
  }

  const size_t batch = images.size();
  const size_t channels = 3;
  const size_t height = images[0].rows;
  const size_t width = images[0].cols;
  const size_t image_size = channels * height * width;
  const size_t required = batch * image_size;

  packed.shape = {batch, channels, height, width};
  packed.data.resize(required);

  std::array<cv::Mat, 3> channel_planes;
  const size_t plane_elements = height * width;
  const size_t plane_bytes = plane_elements * sizeof(float);
  for (auto & plane : channel_planes) {
    plane.create(static_cast<int>(height), static_cast<int>(width), CV_32F);
  }

  for (size_t b = 0; b < batch; ++b) {
    const cv::Mat & img = images[b];
    if (img.channels() != 3 || img.type() != CV_32FC3) {
      throw std::runtime_error("Preprocessed image must be CV_32FC3");
    }

    cv::split(img, channel_planes.data());
    float * batch_base = packed.data.data() + b * image_size;
    for (size_t c = 0; c < channels; ++c) {
      const float * src_ptr = channel_planes[c].ptr<float>();
      float * dst_ptr = batch_base + c * plane_elements;
      std::memcpy(dst_ptr, src_ptr, plane_bytes);
    }
  }

  return packed;
}

}  // namespace deep_object_detection

