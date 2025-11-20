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

        // Try TensorRT as fallback
        try {
          auto gpu_plugin = std::make_shared<deep_ort_gpu_backend::OrtGpuBackendPlugin>(
            device_id_, deep_ort_gpu_backend::GpuExecutionProvider::TENSORRT);
          backend_executor_ = gpu_plugin->get_inference_executor();
          backend_allocator_ = gpu_plugin->get_allocator();
          std::cout << "Initialized GPU backend (ORT TensorRT) on device " << device_id_ << std::endl;
        } catch (const std::exception & tensorrt_e) {
          std::cerr << "Failed to initialize TensorRT backend: " << tensorrt_e.what() << std::endl;
          throw std::runtime_error("Failed to initialize any GPU backend");
        }
        // }
      } break;

      default:
        throw std::runtime_error("Unsupported backend type");
    }

    // Verify backend components are valid
    if (!backend_executor_ || !backend_allocator_) {
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
  RCLCPP_INFO(
    rclcpp::get_logger("OrtBackendInference"), "Image count check passed, processing %zu images", images.size());

  if (images.size() > static_cast<size_t>(config_.max_batch_size)) {
    throw std::runtime_error(
      "Batch size " + std::to_string(images.size()) + " exceeds maximum " + std::to_string(config_.max_batch_size));
  }
  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Batch size check passed");

  std::lock_guard<std::mutex> lock(inference_mutex_);
  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Mutex lock acquired");

  try {
    RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Starting image size collection...");
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
    RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Image size collection completed");

    RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Starting image preprocessing...");
    // Preprocess images and convert to tensor
    std::vector<cv::Mat> preprocessed_images;
    preprocessed_images.reserve(images.size());

    for (size_t i = 0; i < images.size(); ++i) {
      RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Preprocessing image %zu...", i);
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
    RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Image preprocessing completed");

    RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Converting images to tensor...");
    // Convert to backend tensor
    deep_ros::Tensor input_tensor;
    try {
      input_tensor = convertImagesToTensor(preprocessed_images);
      std::ostringstream shape_stream;
      shape_stream << "[";
      for (size_t i = 0; i < input_tensor.shape().size(); ++i) {
        shape_stream << input_tensor.shape()[i];
        if (i < input_tensor.shape().size() - 1) shape_stream << ", ";
      }
      shape_stream << "]";
      RCLCPP_INFO(
        rclcpp::get_logger("OrtBackendInference"),
        "Tensor conversion completed. Tensor shape: %s",
        shape_stream.str().c_str());
    } catch (const std::exception & e) {
      RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Failed to convert images to tensor: %s", e.what());
      throw;
    }

    RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Starting GPU inference...");
    // Run inference
    deep_ros::Tensor output_tensor;
    try {
      output_tensor = backend_executor_->run_inference(input_tensor);
      RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "GPU inference completed successfully");
    } catch (const std::exception & e) {
      RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "GPU inference failed: %s", e.what());
      throw;
    }

    RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Starting post-processing...");
    // Post-process results
    auto results = postprocessOutput(output_tensor, original_sizes);
    RCLCPP_INFO(
      rclcpp::get_logger("OrtBackendInference"), "Post-processing completed. Found %zu result sets", results.size());

    RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "=== EXITING inferBatch SUCCESSFULLY ===");
    return results;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"), "Exception in inferBatch: %s", e.what());
    throw std::runtime_error("Inference failed: " + std::string(e.what()));
  }
}

// deep_ros::Tensor OrtBackendInference::convertImagesToTensor(const std::vector<cv::Mat> & images)
// {
//   RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "=== ENTERING convertImagesToTensor ===");

//   if (images.empty()) {
//     throw std::runtime_error("No images to convert");
//   }
//   RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"),
//     "Processing %zu images for tensor conversion", images.size());

//   // Define tensor shape: [batch_size, channels, height, width]
//   std::vector<size_t> shape = {
//     images.size(),
//     static_cast<size_t>(images[0].channels()),
//     static_cast<size_t>(config_.input_height),
//     static_cast<size_t>(config_.input_width)
//   };

//   std::ostringstream shape_stream;
//   shape_stream << "[";
//   for (size_t i = 0; i < shape.size(); ++i) {
//     shape_stream << shape[i];
//     if (i < shape.size() - 1) shape_stream << ", ";
//   }
//   shape_stream << "]";
//   RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"),
//     "Tensor shape will be: %s", shape_stream.str().c_str());

//   // Calculate total memory needed
//   size_t total_elements = 1;
//   for (auto dim : shape) {
//     total_elements *= dim;
//   }
//   size_t total_bytes = total_elements * sizeof(float);
//   RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"),
//     "Tensor will need %zu elements (%zu bytes)", total_elements, total_bytes);

//   RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Creating tensor with backend allocator...");
//   // Create tensor using backend allocator
//   deep_ros::Tensor tensor;
//   try {
//     if (!backend_allocator_) {
//       throw std::runtime_error("Backend allocator is null");
//     }
//     RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Backend allocator is valid, creating tensor...");

//     tensor = deep_ros::Tensor(shape, deep_ros::DataType::FLOAT32, backend_allocator_);
//     RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Tensor created successfully");

//     // Verify tensor data pointer
//     if (!tensor.data()) {
//       throw std::runtime_error("Tensor data pointer is null after creation");
//     }
//     RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"),
//       "Tensor data pointer is valid: %p", tensor.data());

//   } catch (const std::exception & e) {
//     RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"),
//       "Failed to create tensor: %s", e.what());
//     throw;
//   }

//   RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Starting data copy to tensor...");
//   // Copy image data to tensor
//   float * tensor_data = static_cast<float *>(tensor.data());
//   size_t image_size = config_.input_height * config_.input_width * images[0].channels();

//   for (size_t i = 0; i < images.size(); ++i) {
//     RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Copying image %zu data...", i);
//     const cv::Mat & img = images[i];
//     int depth = img.depth();
//     if (depth != CV_32F) {
//       RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"),
//         "Image must be float type after preprocessing. Got depth: %d, expected: %d (CV_32F)",
//         depth, CV_32F);
//       throw std::runtime_error("Image must be float type after preprocessing");
//     }

//     RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"),
//       "Image %zu validation passed, copying pixel data...", i);

//     // Copy image data (OpenCV uses HWC format, we need CHW for most models)
//     try {
//       for (int c = 0; c < img.channels(); ++c) {
//         for (int h = 0; h < img.rows; ++h) {
//           for (int w = 0; w < img.cols; ++w) {
//             size_t tensor_idx = i * image_size +
//                                c * img.rows * img.cols +
//                                h * img.cols + w;

//             // Bounds check
//             if (tensor_idx >= total_elements) {
//               throw std::runtime_error("Tensor index out of bounds: " +
//                 std::to_string(tensor_idx) + " >= " + std::to_string(total_elements));
//             }

//             tensor_data[tensor_idx] = img.at<cv::Vec3f>(h, w)[c];
//           }
//         }
//       }
//       RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Image %zu data copy completed", i);
//     } catch (const std::exception & e) {
//       RCLCPP_ERROR(rclcpp::get_logger("OrtBackendInference"),
//         "Failed to copy image %zu data: %s", i, e.what());
//       throw;
//     }
//   }

//   RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "All image data copied successfully");
//   RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "=== EXITING convertImagesToTensor SUCCESSFULLY ===");
//   return tensor;
// }

cv::Mat OrtBackendInference::preprocessImage(const cv::Mat & image)
{
  cv::Mat processed;

  // Resize to model input size
  cv::resize(image, processed, cv::Size(config_.input_width, config_.input_height));

  // Convert BGR to RGB if needed
  if (processed.channels() == 3) {
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
  }

  // Convert to float and normalize
  processed.convertTo(processed, CV_32F, 1.0 / 255.0);

  return processed;
}

deep_ros::Tensor OrtBackendInference::convertImagesToTensor(const std::vector<cv::Mat> & images)
{
  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "=== ENTERING convertImagesToTensor ===");

  if (images.empty()) {
    throw std::runtime_error("No images to convert");
  }
  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Processing %zu images for tensor conversion", images.size());

  // Define tensor shape: [batch_size, channels, height, width]
  std::vector<size_t> shape = {
    images.size(),
    static_cast<size_t>(images[0].channels()),
    static_cast<size_t>(config_.input_height),
    static_cast<size_t>(config_.input_width)};

  std::ostringstream shape_stream;
  shape_stream << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_stream << shape[i];
    if (i < shape.size() - 1) shape_stream << ", ";
  }
  shape_stream << "]";
  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Tensor shape will be: %s", shape_stream.str().c_str());

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

  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Creating CPU buffer for data preparation...");

  // Step 1: Create CPU buffer to prepare data
  std::vector<float> cpu_data(total_elements);
  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "CPU buffer created with %zu elements", cpu_data.size());

  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Starting data copy to CPU buffer...");

  // Step 2: Copy image data to CPU buffer
  size_t image_size = config_.input_height * config_.input_width * images[0].channels();

  for (size_t i = 0; i < images.size(); ++i) {
    RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Copying image %zu data to CPU buffer...", i);
    const cv::Mat & img = images[i];
    int depth = img.depth();
    if (depth != CV_32F) {
      RCLCPP_ERROR(
        rclcpp::get_logger("OrtBackendInference"),
        "Image must be float type after preprocessing. Got depth: %d, expected: %d (CV_32F)",
        depth,
        CV_32F);
      throw std::runtime_error("Image must be float type after preprocessing");
    }

    RCLCPP_INFO(
      rclcpp::get_logger("OrtBackendInference"), "Image %zu validation passed, copying pixel data to CPU buffer...", i);

    // Copy image data (OpenCV uses HWC format, we need CHW for most models)
    try {
      for (int c = 0; c < img.channels(); ++c) {
        for (int h = 0; h < img.rows; ++h) {
          for (int w = 0; w < img.cols; ++w) {
            size_t tensor_idx = i * image_size + c * img.rows * img.cols + h * img.cols + w;

            // Bounds check
            if (tensor_idx >= total_elements) {
              throw std::runtime_error(
                "Tensor index out of bounds: " + std::to_string(tensor_idx) + " >= " + std::to_string(total_elements));
            }

            // Safe CPU memory access
            cpu_data[tensor_idx] = img.at<cv::Vec3f>(h, w)[c];
          }
        }
      }
      RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Image %zu data copy to CPU buffer completed", i);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        rclcpp::get_logger("OrtBackendInference"), "Failed to copy image %zu data to CPU buffer: %s", i, e.what());
      throw;
    }
  }

  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "All image data copied to CPU buffer successfully");

  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Creating CPU tensor with simple CPU allocator...");

  // Step 3: Create CPU tensor using simple CPU allocator from the GPU backend factory
  // This ensures we use a consistent CPU allocator without creating CPU backend plugin
  auto cpu_allocator = deep_ort_gpu_backend::get_simple_cpu_allocator();
  deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, cpu_allocator);

  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "CPU tensor created successfully");

  // Step 4: Copy data directly to CPU tensor (no GPU transfers)
  std::memcpy(tensor.data(), cpu_data.data(), total_bytes);

  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "Data successfully copied to CPU tensor");
  RCLCPP_INFO(rclcpp::get_logger("OrtBackendInference"), "=== EXITING convertImagesToTensor SUCCESSFULLY ===");
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
