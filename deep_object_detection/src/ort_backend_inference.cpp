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
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
// Backend plugin headers
#include <deep_ort_backend_plugin/ort_backend_plugin.hpp>
#include <deep_ort_gpu_backend_plugin/ort_gpu_backend_plugin.hpp>
#include <deep_ort_gpu_backend_plugin/ort_gpu_memory_allocator.hpp>
#include <rclcpp/rclcpp.hpp>

namespace deep_object_detection
{

OrtBackendInference::OrtBackendInference(const InferenceConfig & config, BackendType backend_type, int device_id)
: config_(config)
, backend_type_(backend_type)
, device_id_(device_id)
, initialized_(false)
{}

OrtBackendInference::~OrtBackendInference()
{
  // Backend components are automatically cleaned up via shared_ptr
}

bool OrtBackendInference::initialize()
{
  std::lock_guard<std::mutex> lock(inference_mutex_);

  if (initialized_) {
    return true;
  }

  try {
    // Initialize backend based on type
    switch (backend_type_) {
      case BackendType::CPU: {
        auto cpu_plugin = std::make_shared<deep_ort_backend::OrtBackendPlugin>();
        backend_executor_ = cpu_plugin->get_inference_executor();
        backend_allocator_ = cpu_plugin->get_allocator();
        std::cout << "Initialized CPU backend (ORT)" << std::endl;
      } break;

      case BackendType::GPU: {
        // Try CUDA first, fallback to TensorRT if needed
        // try {
        //   auto gpu_plugin = std::make_shared<deep_ort_gpu_backend::OrtGpuBackendPlugin>(
        //     device_id_, deep_ort_gpu_backend::GpuExecutionProvider::CUDA);
        //   backend_executor_ = gpu_plugin->get_inference_executor();
        //   backend_allocator_ = gpu_plugin->get_allocator();
        //   std::cout << "Initialized GPU backend (ORT CUDA) on device " << device_id_ << std::endl;
        // } catch (const std::exception & e) {
        //   std::cerr << "Failed to initialize CUDA backend: " << e.what() << std::endl;

        // Try CUDA first for testing (temporarily disabled TensorRT due to CUDA driver mismatch)
        try {
          auto gpu_plugin = std::make_shared<deep_ort_gpu_backend::OrtGpuBackendPlugin>(
            device_id_, deep_ort_gpu_backend::GpuExecutionProvider::TENSORRT);
          backend_executor_ = gpu_plugin->get_inference_executor();
          std::cout << "Initialized GPU backend (ORT CUDA) on device " << device_id_ << std::endl;
        } catch (const std::exception & e) {
          std::cerr << "Failed to initialize CUDA backend due to driver version mismatch: " << e.what() << std::endl;
          std::cerr << "Falling back to CPU backend for now..." << std::endl;
          auto cpu_plugin = std::make_shared<deep_ort_backend::OrtBackendPlugin>();
          backend_executor_ = cpu_plugin->get_inference_executor();
          std::cout << "Initialized CPU backend (ORT) as fallback" << std::endl;
        }
        // }
      } break;

      default:
        throw std::runtime_error("Unsupported backend type");
    }

    // Verify backend components are valid
    if (!backend_executor_) {
      throw std::runtime_error("Failed to obtain backend components");
    }

    // Load the model
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
  auto batch_results = inferBatch({image});
  if (batch_results.empty()) {
    return {};
  }
  return batch_results[0];
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
  if (images.size() > static_cast<size_t>(config_.max_batch_size)) {
    throw std::runtime_error(
      "Batch size " + std::to_string(images.size()) + " exceeds maximum " + std::to_string(config_.max_batch_size));
  }

  std::lock_guard<std::mutex> lock(inference_mutex_);

  try {
    // Store original image sizes for post-processing
    std::vector<cv::Size> original_sizes;
    original_sizes.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
      const auto & img = images[i];
      RCLCPP_INFO(
        rclcpp::get_logger("OrtBackendInference"),
        "Image %zu size: %dx%d, channels: %d",
        i,
        img.cols,
        img.rows,
        img.channels());

      // Check if image is valid
      if (img.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Image %zu is empty!", i);
        throw std::runtime_error("Image " + std::to_string(i) + " is empty");
      }

      original_sizes.push_back(img.size());
    }

    // Preprocess images and convert to tensor
    std::vector<cv::Mat> preprocessed_images;
    preprocessed_images.reserve(images.size());

    for (size_t i = 0; i < images.size(); ++i) {
      try {
        cv::Mat processed = preprocessImage(images[i]);
        RCLCPP_INFO(
          rclcpp::get_logger("OrtBackendInference"),
          "Processed image %zu size: %dx%d, channels: %d, type: %d",
          i,
          processed.cols,
          processed.rows,
          processed.channels(),
          processed.type());
        preprocessed_images.push_back(processed);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Failed to preprocess image %zu: %s", i, e.what());
        throw;
      }
    }

    // Convert to backend tensor
    deep_ros::Tensor input_tensor;
    try {
      input_tensor = convertImagesToTensor(preprocessed_images);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Failed to convert images to tensor: %s", e.what());
      throw;
    }

    // Run inference
    deep_ros::Tensor output_tensor;
    try {
      // Debug backend executor reuse
      static void * last_backend_ptr = nullptr;
      void * current_backend_ptr = backend_executor_.get();
      if (last_backend_ptr != current_backend_ptr) {
        RCLCPP_WARN(
          rclcpp::get_logger("OrtBackendInference"),
          "🔄 NEW BACKEND EXECUTOR DETECTED! Previous: %p, Current: %p",
          last_backend_ptr,
          current_backend_ptr);
        RCLCPP_WARN(rclcpp::get_logger("OrtBackendInference"), "   This should only happen once at startup!");
        last_backend_ptr = current_backend_ptr;
      }

      output_tensor = backend_executor_->run_inference(input_tensor);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "GPU inference failed: %s", e.what());
      throw;
    }
    // Post-process results
    auto results = postprocessOutput(output_tensor, original_sizes);
    return results;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Exception in inferBatch: %s", e.what());
    throw std::runtime_error("Inference failed: " + std::string(e.what()));
  }
}

cv::Mat OrtBackendInference::preprocessImage(const cv::Mat & image)
{
  cv::Mat processed;

  // Resize to model input size
  cv::resize(image, processed, cv::Size(config_.input_width, config_.input_height));

  // Convert BGR to RGB if needed
  if (processed.channels() == 3) {
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
  } else if (processed.channels() == 4) {
    cv::cvtColor(processed, processed, cv::COLOR_BGRA2RGB);
  } else if (processed.channels() == 1) {
    cv::cvtColor(processed, processed, cv::COLOR_GRAY2RGB);
  }

  // Ensure processed image is strictly 3-channel RGB before proceeding
  if (processed.channels() != 3) {
    throw std::runtime_error("Processed image must have 3 channels (RGB), but has " + std::to_string(processed.channels()));
  }

  // Convert to float and normalize
  processed.convertTo(processed, CV_32F, 1.0 / 255.0);

  return processed;
}

deep_ros::Tensor OrtBackendInference::convertImagesToTensor(const std::vector<cv::Mat> & images)
{
  if (images.empty()) {
    throw std::runtime_error("No images to convert");
  }
  // Define tensor shape: [batch_size, channels, height, width]
  std::vector<size_t> shape = {
    images.size(),
    static_cast<size_t>(images[0].channels()),
    static_cast<size_t>(config_.input_height),
    static_cast<size_t>(config_.input_width)};

  // Calculate total memory needed
  size_t total_elements = 1;
  for (auto dim : shape) {
    total_elements *= dim;
  }
  size_t total_bytes = total_elements * sizeof(float);
  RCLCPP_INFO(
    rclcpp::get_logger("OrtBackendInference"),
    "Tensor will need %zu elements (%zu bytes)",
    total_elements,
    total_bytes);

  // Step 1: Create CPU tensor using simple CPU allocator (64-byte aligned)
  auto cpu_allocator = deep_ort_gpu_backend::get_ort_gpu_cpu_allocator();
  deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, cpu_allocator);

  // Step 2: Direct copy image data to tensor memory (no intermediate buffers!)
  float * tensor_data = static_cast<float *>(tensor.data());
  size_t image_size = config_.input_height * config_.input_width * images[0].channels();

  for (size_t i = 0; i < images.size(); ++i) {
    const cv::Mat & img = images[i];

    if (img.depth() != CV_32F) {
      throw std::runtime_error("Image must be float type after preprocessing");
    }

    // Direct copy to tensor memory (HWC -> CHW conversion)
    if (img.channels() != 3) {
      throw std::runtime_error("Image must have 3 channels for HWC->CHW conversion");
    }

    // Get direct access to raw data pointer
    const float* img_data = reinterpret_cast<const float*>(img.data);

    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < img.rows; ++h) {
        for (int w = 0; w < img.cols; ++w) {
          size_t tensor_idx = i * image_size + c * img.rows * img.cols + h * img.cols + w;
          // img.at<cv::Vec3f>(h, w)[c] is slow and unsafe if not exactly Vec3f
          // Using raw pointer arithmetic assuming packed 3-channel float (HWC)
          // Index in HWC image: (h * cols + w) * channels + c
          size_t pixel_idx = (h * img.cols + w) * 3 + c;
          tensor_data[tensor_idx] = img_data[pixel_idx];
        }
      }
    }
  }

  return tensor;
}

std::vector<std::vector<Detection>> OrtBackendInference::postprocessOutput(
  const deep_ros::Tensor & output_tensor, const std::vector<cv::Size> & original_sizes)
{
  std::vector<std::vector<Detection>> results;
  results.reserve(original_sizes.size());

  // Get tensor data and shape
  const float * output_data = static_cast<const float *>(output_tensor.data());
  const std::vector<size_t> & output_shape = output_tensor.shape();

  // Validate output data
  if (output_data == nullptr) {
    RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Output tensor data is null");
    return results;
  }

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
  
  // Adaptive shape detection: YOLOv8 usually outputs [batch, 84, 8400]
  // But some exports or models might use [batch, 8400, 84]
  // We determine the format based on which dimension is closer to the number of classes + 4
  size_t dim1 = output_shape[1];
  size_t dim2 = output_shape[2];
  
  bool is_transposed = false; // transposed means [batch, detections, values]
  size_t num_values = 0;
  size_t num_detections = 0;
  
  // Heuristic: The dimension closer to (4 + 80) is likely the values dimension
  // This assumes we have roughly 80 classes. If both are large, we default to standard YOLOv8 layout.
  if (dim1 < dim2 && dim1 < 200) {
      // Standard YOLOv8: [batch, values, detections]
      num_values = dim1;
      num_detections = dim2;
      is_transposed = false;
  } else {
      // Transposed: [batch, detections, values]
      num_detections = dim1;
      num_values = dim2;
      is_transposed = true;
      RCLCPP_DEBUG(rclcpp::get_logger("OrtBackendInference"), "Detected transposed output format [batch, detections, values]");
  }

  if (image_index >= batch_size) {
    RCLCPP_ERROR(
      rclcpp::get_logger("OrtBackendInference"), "Image index %zu exceeds batch size %zu", image_index, batch_size);
    return detections;
  }
  
  // Defensive: ensure we have at least 4 values (bbox)
  if (num_values < 4) {
    RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Not enough values per detection: %zu", num_values);
    return detections;
  }

  // Calculate stride and offset based on layout
  size_t image_stride = num_values * num_detections;
  size_t image_offset = image_index * image_stride;

  // Calculate total elements from output shape to ensure we don't read past buffer
  size_t total_elements = 1;
  for (size_t dim : output_shape) {
    total_elements *= dim;
  }

  for (size_t i = 0; i < num_detections; ++i) {
    float raw_cx, raw_cy, raw_w, raw_h;
    
    // Bounds check for safety (though logic below should be safe if strides are correct)
    if (image_offset + (is_transposed ? (i * num_values + 3) : (3 * num_detections + i)) >= total_elements) {
        break; 
    }

    if (!is_transposed) {
        // Standard [batch, values, detections]
        // values are rows, detections are columns
        raw_cx = output_data[image_offset + 0 * num_detections + i];
        raw_cy = output_data[image_offset + 1 * num_detections + i];
        raw_w = output_data[image_offset + 2 * num_detections + i];
        raw_h = output_data[image_offset + 3 * num_detections + i];
    } else {
        // Transposed [batch, detections, values]
        // detections are rows, values are columns
        size_t det_offset = image_offset + i * num_values;
        raw_cx = output_data[det_offset + 0];
        raw_cy = output_data[det_offset + 1];
        raw_w = output_data[det_offset + 2];
        raw_h = output_data[det_offset + 3];
    }

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
    
    // Check if we have an objectness score (some models like YOLOv5/v7 might have it at index 4)
    // Heuristic: if num_values == 85 (4 bbox + 1 objectness + 80 classes), index 4 is objectness.
    // YOLOv8 usually has 84 (4 bbox + 80 classes) and no objectness.
    bool has_objectness = (num_values == (4 + 1 + 80)); // Assuming 80 classes
    float objectness = 1.0f;
    size_t class_start_idx = 4;
    
    if (has_objectness) {
        if (!is_transposed) {
             objectness = output_data[image_offset + 4 * num_detections + i];
        } else {
             objectness = output_data[image_offset + i * num_values + 4];
        }
        class_start_idx = 5;
    }
    
    if (objectness < config_.confidence_threshold) continue;

    for (size_t c = class_start_idx; c < num_values; ++c) { 
      float cls_score;
      if (!is_transposed) {
          cls_score = output_data[image_offset + c * num_detections + i];
      } else {
          cls_score = output_data[image_offset + i * num_values + c];
      }
      
      if (cls_score > max_score) {
        max_score = cls_score;
        best_class_id = static_cast<int>(c - class_start_idx);
      }
    }

    // In YOLOv8, confidence is usually just the class score. If objectness exists, combine them.
    float confidence = max_score * objectness;

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
