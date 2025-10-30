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

#include "deep_object_detection/ort_backend_inference.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace deep_object_detection
{

OrtBackendInference::OrtBackendInference(const InferenceConfig & config)
: config_(config)
, initialized_(false)
{
  backend_executor_ = std::make_unique<deep_ort_backend::OrtBackendExecutor>();
}

OrtBackendInference::~OrtBackendInference() = default;

bool OrtBackendInference::initialize()
{
  try {
    // Load the ONNX model using the ORT backend executor
    if (!backend_executor_->load_model(config_.model_path)) {
      RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Failed to load model: %s", config_.model_path.c_str());
      return false;
    }

    initialized_ = true;
    RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "ORT Backend inference initialized successfully");
    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Failed to initialize: %s", e.what());
    return false;
  }
}

std::vector<Detection> OrtBackendInference::infer(const cv::Mat & image)
{
  return inferBatch({image})[0];
}

std::vector<std::vector<Detection>> OrtBackendInference::inferBatch(const std::vector<cv::Mat> & images)
{
  if (!initialized_) {
    RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Inference engine not initialized");
    return {};
  }

  if (images.empty()) {
    return {};
  }

  std::lock_guard<std::mutex> lock(inference_mutex_);

  try {
    // Store original sizes for post-processing
    std::vector<cv::Size> original_sizes;
    std::vector<cv::Mat> processed_images;

    original_sizes.reserve(images.size());
    processed_images.reserve(images.size());

    // Preprocess images
    for (const auto & image : images) {
      original_sizes.push_back(image.size());
      processed_images.push_back(preprocessImage(image));
    }

    // Convert images to tensor format expected by ORT backend
    deep_ros::Tensor input_tensor = convertImagesToTensor(processed_images);

    // Debug: Log input tensor shape
    const auto & input_shape = input_tensor.shape();
    RCLCPP_DEBUG(
      rclcpp::get_logger("OrtBackendInference"),
      "Input tensor shape: [%zu, %zu, %zu, %zu]",
      input_shape[0],
      input_shape[1],
      input_shape[2],
      input_shape[3]);

    // Run inference using ORT backend executor
    deep_ros::Tensor output_tensor = backend_executor_->run_inference(input_tensor);
    // Post-process the output tensor
    return postprocessOutput(output_tensor, original_sizes);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Inference failed: %s", e.what());
    return {};
  }
}

cv::Mat OrtBackendInference::preprocessImage(const cv::Mat & image)
{
  cv::Mat tmp = image;

  // Ensure 3-channel BGR input
  if (tmp.channels() == 1) {
    cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);
  } else if (tmp.channels() == 4) {
    cv::cvtColor(tmp, tmp, cv::COLOR_BGRA2BGR);
  }

  cv::Mat processed;
  // Resize image to model input size
  cv::resize(tmp, processed, cv::Size(config_.input_width, config_.input_height));

  // Convert to float and normalize to [0, 1], ensure 3-channel float
  processed.convertTo(processed, CV_32F, 1.0 / 255.0);

  // Guarantee type is CV_32FC3
  if (processed.type() != CV_32FC3) {
    processed = processed.reshape(3);
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);
  }

  return processed;
}

deep_ros::Tensor OrtBackendInference::convertImagesToTensor(const std::vector<cv::Mat> & images)
{
  size_t batch_size = images.size();

  // Create tensor shape: [batch_size, channels, height, width]
  std::vector<size_t> shape = {
    batch_size, 3, static_cast<size_t>(config_.input_height), static_cast<size_t>(config_.input_width)};
  size_t total_elements = std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<size_t>());

  // Allocate heap buffer that will be owned by the deep_ros::Tensor
  float * tensor_ptr = new float[total_elements];

  // Convert OpenCV images to tensor format (NCHW: batch, channel, height, width)
  for (size_t b = 0; b < batch_size; ++b) {
    const cv::Mat & image = images[b];

    // Ensure image is in the expected format CV_32FC3
    if (image.type() != CV_32FC3) {
      // defensive conversion if caller didn't produce CV_32FC3
      cv::Mat tmp;
      image.convertTo(tmp, CV_32F, 1.0 / 255.0);
      if (tmp.channels() == 1) cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);
      if (tmp.channels() == 4) cv::cvtColor(tmp, tmp, cv::COLOR_BGRA2BGR);
      if (tmp.type() != CV_32FC3) tmp.convertTo(tmp, CV_32F);
      // use tmp for pixel reads below by copying back to image_ref
      // create a temporary reference variable
      // but to keep code simple, operate on tmp via pointer access below
      const cv::Mat & use_img = tmp;
      size_t image_offset = b * 3 * config_.input_height * config_.input_width;
      for (int c = 0; c < 3; ++c) {
        int cv_channel = 2 - c;  // BGR -> RGB
        for (int h = 0; h < config_.input_height; ++h) {
          const float * row_ptr = use_img.ptr<float>(h);
          for (int w = 0; w < config_.input_width; ++w) {
            float pixel_value = row_ptr[w * 3 + cv_channel];
            size_t tensor_idx =
              image_offset + c * config_.input_height * config_.input_width + h * config_.input_width + w;
            tensor_ptr[tensor_idx] = pixel_value;
          }
        }
      }
      continue;
    }

    size_t image_offset = b * 3 * config_.input_height * config_.input_width;

    // Convert from HWC (OpenCV) to CHW (tensor) format
    for (int c = 0; c < 3; ++c) {
      int cv_channel = 2 - c;  // BGR -> RGB conversion
      for (int h = 0; h < config_.input_height; ++h) {
        const cv::Vec3f * row_ptr = image.ptr<cv::Vec3f>(h);
        for (int w = 0; w < config_.input_width; ++w) {
          float pixel_value = row_ptr[w][cv_channel];

          size_t tensor_idx =
            image_offset + c * config_.input_height * config_.input_width + h * config_.input_width + w;
          tensor_ptr[tensor_idx] = pixel_value;
        }
      }
    }
  }

  // Create tensor with the data (tensor takes ownership of tensor_ptr)
  return deep_ros::Tensor(static_cast<void *>(tensor_ptr), shape, deep_ros::DataType::FLOAT32);
}

std::vector<std::vector<Detection>> OrtBackendInference::postprocessOutput(
  const deep_ros::Tensor & output_tensor, const std::vector<cv::Size> & original_sizes)
{
  std::vector<std::vector<Detection>> results;
  results.reserve(original_sizes.size());

  // Get tensor data and shape
  const float * output_data = static_cast<const float *>(output_tensor.data());
  const std::vector<size_t> & output_shape = output_tensor.shape();
  // Basic sanity check on output shape
  if (output_shape.size() < 2) {
    RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Unexpected output tensor shape");
    return results;
  }

  // Enhanced debugging for dynamic shapes
  RCLCPP_DEBUG(rclcpp::get_logger("OrtBackendInference"), "=== TENSOR DEBUG INFO ===");
  RCLCPP_DEBUG(rclcpp::get_logger("OrtBackendInference"), "Shape dimensions: %zu", output_shape.size());

  std::ostringstream ss;
  ss << "Shape: [";
  for (size_t i = 0; i < output_shape.size(); ++i) {
    ss << output_shape[i];
    if (i < output_shape.size() - 1) ss << ", ";
  }
  ss << "]";
  RCLCPP_DEBUG(rclcpp::get_logger("OrtBackendInference"), "%s", ss.str().c_str());

  // Calculate total elements
  size_t total_elements = 1;
  bool has_zero_dim = false;
  for (auto dim : output_shape) {
    if (dim == 0) {
      has_zero_dim = true;
      RCLCPP_ERROR(
        rclcpp::get_logger("OrtBackendInference"), "Found zero dimension in output shape - model has dynamic shapes!");
      break;
    }
    total_elements *= dim;
  }

  if (has_zero_dim) {
    RCLCPP_ERROR(
      rclcpp::get_logger("OrtBackendInference"),
      "Cannot process tensor with dynamic/zero dimensions. Model needs fixed input shapes or executor needs fixing.");
    return results;
  }

  RCLCPP_DEBUG(rclcpp::get_logger("OrtBackendInference"), "Total elements: %zu", total_elements);
  // Process detections for each image in the batch
  for (size_t i = 0; i < original_sizes.size(); ++i) {
    auto detections = processDetectionsForImage(output_data, output_shape, i, original_sizes[i]);
    results.push_back(applyNMS(detections));
  }

  return results;
}

std::vector<Detection> OrtBackendInference::processDetectionsForImage(
  const float * output_data,
  const std::vector<size_t> & output_shape,
  size_t image_index,
  const cv::Size & original_size)
{
  std::vector<Detection> detections;

  // YOLOv8 output format: [batch_size, num_values, num_detections]
  // where num_values = 4 (bbox) + num_classes (80 for COCO)
  if (output_shape.size() != 3) {
    RCLCPP_ERROR(
      rclcpp::get_logger("OrtBackendInference"), "Unexpected output shape dimensions: %zu", output_shape.size());
    return detections;
  }

  size_t batch_size = output_shape[0];
  size_t num_values = output_shape[1];  // 84 (4 bbox + 80 classes)
  size_t num_detections = output_shape[2];  // 8400 anchors

  if (image_index >= batch_size) {
    RCLCPP_ERROR(
      rclcpp::get_logger("OrtBackendInference"), "Image index %zu exceeds batch size %zu", image_index, batch_size);
    return detections;
  }

  // Offset to this image's data in the output tensor
  size_t image_offset = image_index * num_values * num_detections;

  for (size_t i = 0; i < num_detections; ++i) {
    // For transposed format [batch, values, detections], each value spans all detections
    // So detection i has its bbox at: [0][i], [1][i], [2][i], [3][i]
    // And its classes at: [4][i], [5][i], ..., [83][i]

    // Defensive: ensure we have at least 4 values (bbox)
    if (num_values < 4) {
      RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Not enough values per detection: %zu", num_values);
      break;
    }

    // Read raw values for transposed format [batch, values, detections]
    // For detection i: bbox coords are at [0][i], [1][i], [2][i], [3][i]
    float raw_cx = output_data[image_offset + 0 * num_detections + i];
    float raw_cy = output_data[image_offset + 1 * num_detections + i];
    float raw_w = output_data[image_offset + 2 * num_detections + i];
    float raw_h = output_data[image_offset + 3 * num_detections + i];

    // Determine whether values are normalized (<=1) or in pixels (likely >1).
    // If normalized -> multiply by original size. If >1 but likely in input-pixel space, scale to original.
    float cx, cy, w, h;
    bool normalized = (raw_cx <= 1.0f && raw_cy <= 1.0f && raw_w <= 1.0f && raw_h <= 1.0f);

    if (normalized) {
      cx = raw_cx * static_cast<float>(original_size.width);
      cy = raw_cy * static_cast<float>(original_size.height);
      w = raw_w * static_cast<float>(original_size.width);
      h = raw_h * static_cast<float>(original_size.height);
    } else {
      // assume coords are in input pixel units (relative to config_.input_width/height)
      float scale_x = static_cast<float>(original_size.width) / static_cast<float>(config_.input_width);
      float scale_y = static_cast<float>(original_size.height) / static_cast<float>(config_.input_height);
      cx = raw_cx * scale_x;
      cy = raw_cy * scale_y;
      w = raw_w * scale_x;
      h = raw_h * scale_y;
    }

    // Find class with highest score (YOLOv8 format: no separate objectness score)
    float max_score = 0.0f;
    int best_class_id = -1;
    for (size_t c = 4; c < num_values; ++c) {  // Skip bbox (4), no objectness in YOLOv8
      float cls_score = output_data[image_offset + c * num_detections + i];
      if (cls_score > max_score) {
        max_score = cls_score;
        best_class_id = static_cast<int>(c - 4);
      }
    }

    // In YOLOv8, confidence is just the class score
    float confidence = max_score;

    // Defensive fixes: clamp confidence and validate class id
    if (!std::isfinite(confidence)) {
      continue;
    }
    confidence = std::max(0.0f, std::min(1.0f, confidence));

    if (confidence >= config_.confidence_threshold && best_class_id >= 0) {
      Detection det;
      det.x = cx - w / 2.0f;  // Convert to top-left format
      det.y = cy - h / 2.0f;
      det.width = w;
      det.height = h;
      det.confidence = confidence;
      det.class_id = best_class_id;
      det.class_name =
        (best_class_id < static_cast<int>(config_.class_names.size())) ? config_.class_names[best_class_id] : "unknown";

      detections.push_back(det);
    }
  }

  return detections;
}

std::vector<Detection> OrtBackendInference::applyNMS(const std::vector<Detection> & detections)
{
  if (detections.empty()) {
    return {};
  }

  // Sort by confidence (descending)
  std::vector<Detection> sorted_detections = detections;
  std::sort(sorted_detections.begin(), sorted_detections.end(), [](const Detection & a, const Detection & b) {
    return a.confidence > b.confidence;
  });

  std::vector<bool> suppressed(sorted_detections.size(), false);
  std::vector<Detection> result;

  for (size_t i = 0; i < sorted_detections.size(); ++i) {
    if (suppressed[i]) continue;

    result.push_back(sorted_detections[i]);

    // Suppress overlapping detections
    for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
      if (suppressed[j]) continue;

      // Calculate IoU
      float x1 = std::max(sorted_detections[i].x, sorted_detections[j].x);
      float y1 = std::max(sorted_detections[i].y, sorted_detections[j].y);
      float x2 = std::min(
        sorted_detections[i].x + sorted_detections[i].width, sorted_detections[j].x + sorted_detections[j].width);
      float y2 = std::min(
        sorted_detections[i].y + sorted_detections[i].height, sorted_detections[j].y + sorted_detections[j].height);

      if (x2 > x1 && y2 > y1) {
        float intersection = (x2 - x1) * (y2 - y1);
        float area1 = sorted_detections[i].width * sorted_detections[i].height;
        float area2 = sorted_detections[j].width * sorted_detections[j].height;
        float iou = intersection / (area1 + area2 - intersection);

        if (iou > config_.nms_threshold) {
          suppressed[j] = true;
        }
      }
    }
  }

  return result;
}

}  // namespace deep_object_detection
