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

#include "deep_object_detection/onnx_inference.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace deep_object_detection
{

ONNXInference::ONNXInference(const InferenceConfig & config)
: config_(config)
, initialized_(false)
{}

ONNXInference::~ONNXInference() = default;

bool ONNXInference::initialize()
{
  try {
    // Create ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");

    // Create session options
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(1);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Setup execution providers
    setupProviders();

    // Load the model
    if (!loadModel()) {
      RCLCPP_ERROR(rclcpp::get_logger("ONNXInference"), "Failed to load ONNX model");
      return false;
    }

    logModelInfo();
    initialized_ = true;

    RCLCPP_INFO(rclcpp::get_logger("ONNXInference"), "ONNX Runtime inference engine initialized successfully");
    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("ONNXInference"), "Failed to initialize ONNX Runtime: %s", e.what());
    return false;
  }
}

void ONNXInference::setupProviders()
{
  if (config_.use_gpu) {
    try {
      // Try to add CUDA execution provider
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id = 0;
      cuda_options.arena_extend_strategy = 1;
      cuda_options.gpu_mem_limit = static_cast<size_t>(2) * 1024 * 1024 * 1024;  // 2GB
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
      cuda_options.do_copy_in_default_stream = 1;

      session_options_->AppendExecutionProvider_CUDA(cuda_options);
      RCLCPP_INFO(rclcpp::get_logger("ONNXInference"), "Using CUDA execution provider");
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        rclcpp::get_logger("ONNXInference"), "Failed to setup CUDA provider, falling back to CPU: %s", e.what());
    }
  }

  // Always add CPU as fallback
  // Note: No explicit CPU provider needed in newer ONNX Runtime versions - it's always available
}

bool ONNXInference::loadModel()
{
  try {
    // Load the ONNX model
    session_ = std::make_unique<Ort::Session>(*env_, config_.model_path.c_str(), *session_options_);

    // Get model input/output info
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input names and shapes
    size_t num_input_nodes = session_->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
      auto input_name = session_->GetInputNameAllocated(i, allocator);
      input_names_.push_back(std::string(input_name.get()));

      auto input_shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      input_shapes_.push_back(input_shape);
    }

    // Get output names and shapes
    size_t num_output_nodes = session_->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
      auto output_name = session_->GetOutputNameAllocated(i, allocator);
      output_names_.push_back(std::string(output_name.get()));

      auto output_shape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      output_shapes_.push_back(output_shape);
    }

    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("ONNXInference"), "Failed to load model: %s", e.what());
    return false;
  }
}

void ONNXInference::logModelInfo()
{
  RCLCPP_INFO(rclcpp::get_logger("ONNXInference"), "Model loaded: %s", config_.model_path.c_str());

  for (size_t i = 0; i < input_names_.size(); i++) {
    std::string shape_str = "[";
    for (size_t j = 0; j < input_shapes_[i].size(); j++) {
      shape_str += std::to_string(input_shapes_[i][j]);
      if (j < input_shapes_[i].size() - 1) shape_str += ", ";
    }
    shape_str += "]";
    RCLCPP_INFO(
      rclcpp::get_logger("ONNXInference"), "Input %zu: %s, shape: %s", i, input_names_[i].c_str(), shape_str.c_str());
  }

  for (size_t i = 0; i < output_names_.size(); i++) {
    std::string shape_str = "[";
    for (size_t j = 0; j < output_shapes_[i].size(); j++) {
      shape_str += std::to_string(output_shapes_[i][j]);
      if (j < output_shapes_[i].size() - 1) shape_str += ", ";
    }
    shape_str += "]";
    RCLCPP_INFO(
      rclcpp::get_logger("ONNXInference"), "Output %zu: %s, shape: %s", i, output_names_[i].c_str(), shape_str.c_str());
  }
}

std::vector<Detection> ONNXInference::infer(const cv::Mat & image)
{
  return inferBatch({image})[0];
}

std::vector<std::vector<Detection>> ONNXInference::inferBatch(const std::vector<cv::Mat> & images)
{
  if (!initialized_) {
    RCLCPP_ERROR(rclcpp::get_logger("ONNXInference"), "Inference engine not initialized");
    return {};
  }

  if (images.empty()) {
    return {};
  }

  std::lock_guard<std::mutex> lock(inference_mutex_);

  try {
    std::vector<std::vector<Detection>> results;

    // Process images in batches
    size_t batch_size = std::min(images.size(), static_cast<size_t>(config_.max_batch_size));

    for (size_t start_idx = 0; start_idx < images.size(); start_idx += batch_size) {
      size_t end_idx = std::min(start_idx + batch_size, images.size());
      size_t current_batch_size = end_idx - start_idx;

      // Prepare input tensor
      std::vector<int64_t> input_shape = {
        static_cast<int64_t>(current_batch_size), 3, config_.input_height, config_.input_width};
      size_t input_tensor_size = current_batch_size * 3 * config_.input_height * config_.input_width;
      std::vector<float> input_tensor_values(input_tensor_size);

      // Preprocess images and fill input tensor
      std::vector<cv::Size> original_sizes;
      for (size_t i = start_idx; i < end_idx; i++) {
        original_sizes.push_back(images[i].size());
        cv::Mat processed = preprocessImage(images[i]);

        // Copy to input tensor (CHW format)
        size_t img_offset = (i - start_idx) * 3 * config_.input_height * config_.input_width;
        for (int c = 0; c < 3; c++) {
          for (int h = 0; h < config_.input_height; h++) {
            for (int w = 0; w < config_.input_width; w++) {
              size_t tensor_idx =
                img_offset + c * config_.input_height * config_.input_width + h * config_.input_width + w;
              input_tensor_values[tensor_idx] = processed.at<cv::Vec3f>(h, w)[c];
            }
          }
        }
      }

      // Create input tensor
      Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      std::vector<Ort::Value> input_tensors;
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size()));

      // Prepare input/output names
      std::vector<const char *> input_names_cstr;
      for (const auto & name : input_names_) {
        input_names_cstr.push_back(name.c_str());
      }
      std::vector<const char *> output_names_cstr;
      for (const auto & name : output_names_) {
        output_names_cstr.push_back(name.c_str());
      }

      // Run inference
      auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr.data(),
        input_tensors.data(),
        1,
        output_names_cstr.data(),
        output_names_cstr.size());

      // Post-process outputs for each image in the batch
      for (size_t i = 0; i < current_batch_size; i++) {
        auto detections = postprocessOutputs(output_tensors, original_sizes[i]);
        results.push_back(applyNMS(detections));
      }
    }

    return results;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("ONNXInference"), "Inference failed: %s", e.what());
    return {};
  }
}

cv::Mat ONNXInference::preprocessImage(const cv::Mat & image)
{
  cv::Mat processed;

  // Resize image
  cv::resize(image, processed, cv::Size(config_.input_width, config_.input_height));

  // Convert to float and normalize to [0, 1]
  processed.convertTo(processed, CV_32F, 1.0 / 255.0);

  return processed;
}

std::vector<Detection> ONNXInference::postprocessOutputs(
  const std::vector<Ort::Value> & outputs, const cv::Size & original_size)
{
  std::vector<Detection> detections;

  if (outputs.empty()) {
    return detections;
  }

  // Assuming YOLOv8-style output: [batch_size, num_detections, 4 + 1 + num_classes]
  // where 4 = bbox coords, 1 = objectness, num_classes = class scores
  auto & output_tensor = outputs[0];
  auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
  auto output_shape = tensor_info.GetShape();

  if (output_shape.size() != 3) {
    RCLCPP_ERROR(rclcpp::get_logger("ONNXInference"), "Unexpected output shape dimensions: %zu", output_shape.size());
    return detections;
  }

  const float * output_data = output_tensor.GetTensorData<float>();
  int num_detections = output_shape[1];
  int num_values = output_shape[2];

  float scale_x = static_cast<float>(original_size.width) / config_.input_width;
  float scale_y = static_cast<float>(original_size.height) / config_.input_height;

  for (int i = 0; i < num_detections; i++) {
    const float * detection = output_data + i * num_values;

    // Extract bbox coordinates (center format)
    float cx = detection[0] * scale_x;
    float cy = detection[1] * scale_y;
    float w = detection[2] * scale_x;
    float h = detection[3] * scale_y;

    // Find class with highest score
    float max_score = 0.0f;
    int best_class_id = -1;
    for (int c = 5; c < num_values; c++) {  // Skip bbox (4) + objectness (1)
      if (detection[c] > max_score) {
        max_score = detection[c];
        best_class_id = c - 5;
      }
    }

    float confidence = detection[4] * max_score;  // objectness * class_score

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

std::vector<Detection> ONNXInference::applyNMS(const std::vector<Detection> & detections)
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

  for (size_t i = 0; i < sorted_detections.size(); i++) {
    if (suppressed[i]) continue;

    result.push_back(sorted_detections[i]);

    // Suppress overlapping detections
    for (size_t j = i + 1; j < sorted_detections.size(); j++) {
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
