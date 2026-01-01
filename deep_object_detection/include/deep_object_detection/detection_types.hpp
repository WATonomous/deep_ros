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

#pragma once

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <std_msgs/msg/header.hpp>

namespace deep_object_detection
{

enum class Provider
{
  TENSORRT,
  CUDA,
  CPU
};

/// @brief Postprocessor type enumeration (deprecated - always uses generic)
enum class PostprocessorType
{
  GENERIC     // Generic auto-detecting postprocessor (always used)
};

/// @brief Bounding box format enumeration
enum class BboxFormat
{
  CXCYWH,     // Center-x, center-y, width, height
  XYXY,       // Top-left x, top-left y, bottom-right x, bottom-right y
  XYWH        // Top-left x, top-left y, width, height
};

/// @brief Normalization type enumeration
enum class NormalizationType
{
  IMAGENET,   // ImageNet mean/std normalization
  SCALE_0_1,  // Simple 0-255 to 0-1 normalization (scale by 1/255)
  CUSTOM,     // Custom mean/std values
  NONE        // No normalization
};

/// @brief Resize method enumeration
enum class ResizeMethod
{
  LETTERBOX,  // Letterbox resize with padding
  RESIZE,     // Simple resize (may distort aspect ratio)
  CROP,       // Center crop
  PAD         // Pad to target size
};

/// @brief Score activation enumeration
enum class ScoreActivation
{
  SIGMOID,    // Sigmoid activation
  SOFTMAX,    // Softmax activation
  NONE        // No activation (raw scores)
};

/// @brief Image metadata for coordinate transformation
struct ImageMeta
{
  int original_width = 0;
  int original_height = 0;
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
};

/// @brief Queued image with header for batch processing
struct QueuedImage
{
  cv::Mat bgr;
  std_msgs::msg::Header header;
};

/// @brief Packed input tensor data
struct PackedInput
{
  std::vector<float> data;
  std::vector<size_t> shape;
};

/// @brief Preprocessing configuration
struct PreprocessingConfig
{
  int input_width = 640;
  int input_height = 640;
  NormalizationType normalization_type = NormalizationType::SCALE_0_1;
  std::vector<float> mean = {0.0f, 0.0f, 0.0f};
  std::vector<float> std = {1.0f, 1.0f, 1.0f};
  ResizeMethod resize_method = ResizeMethod::LETTERBOX;
  bool keep_aspect_ratio = true;
  int pad_value = 114;
  std::string color_format = "bgr";  // "bgr" or "rgb"
};

/// @brief Model metadata configuration
struct ModelMetadata
{
  int num_classes = 80;
  std::string class_names_file;
  BboxFormat bbox_format = BboxFormat::CXCYWH;  // Auto-detected from model output
};

/// @brief Postprocessing configuration
struct PostprocessingConfig
{
  float score_threshold = 0.25f;
  float nms_iou_threshold = 0.45f;
  int max_detections = 300;
};

/// @brief Main detection node parameters
struct DetectionParams
{
  // Model configuration
  std::string model_path;
  ModelMetadata model_metadata;
  
  // Preprocessing configuration
  PreprocessingConfig preprocessing;
  
  // Postprocessing configuration
  PostprocessingConfig postprocessing;
  
  // Topic configuration
  std::string input_image_topic{"/camera/image_raw"};
  std::vector<std::string> camera_topics;
  std::string input_transport{"raw"};
  std::string input_qos_reliability{"best_effort"};
  std::string output_detections_topic{"/detections"};
  
  // Batching configuration
  int batch_size_limit{3};
  int queue_size{10};
  
  // Backend configuration
  std::string preferred_provider{"tensorrt"};
  int device_id{0};
  bool warmup_tensor_shapes{true};
  bool enable_trt_engine_cache{false};
  std::string trt_engine_cache_path{"/tmp/deep_ros_ort_trt_cache"};
  
  // Class names (loaded from file or config)
  std::vector<std::string> class_names;
};

/// @brief Helper to convert string to PostprocessorType (deprecated - always returns GENERIC)
inline PostprocessorType stringToPostprocessorType(const std::string & /*type*/)
{
  return PostprocessorType::GENERIC;  // Always use generic
}

/// @brief Helper to convert string to BboxFormat
inline BboxFormat stringToBboxFormat(const std::string & format)
{
  if (format == "cxcywh" || format == "CXCYWH") {
    return BboxFormat::CXCYWH;
  } else if (format == "xyxy" || format == "XYXY") {
    return BboxFormat::XYXY;
  } else if (format == "xywh" || format == "XYWH") {
    return BboxFormat::XYWH;
  }
  return BboxFormat::CXCYWH;  // Default
}

/// @brief Helper to convert string to NormalizationType
inline NormalizationType stringToNormalizationType(const std::string & type)
{
  if (type == "imagenet" || type == "IMAGENET") {
    return NormalizationType::IMAGENET;
  } else if (type == "scale_0_1" || type == "SCALE_0_1" || type == "yolo" || type == "YOLO") {
    return NormalizationType::SCALE_0_1;  // "yolo" kept for backward compatibility
  } else if (type == "custom" || type == "CUSTOM") {
    return NormalizationType::CUSTOM;
  } else if (type == "none" || type == "NONE") {
    return NormalizationType::NONE;
  }
  return NormalizationType::SCALE_0_1;  // Default
}

/// @brief Helper to convert string to ResizeMethod
inline ResizeMethod stringToResizeMethod(const std::string & method)
{
  if (method == "letterbox" || method == "LETTERBOX") {
    return ResizeMethod::LETTERBOX;
  } else if (method == "resize" || method == "RESIZE") {
    return ResizeMethod::RESIZE;
  } else if (method == "crop" || method == "CROP") {
    return ResizeMethod::CROP;
  } else if (method == "pad" || method == "PAD") {
    return ResizeMethod::PAD;
  }
  return ResizeMethod::LETTERBOX;  // Default
}

/// @brief Helper to convert string to ScoreActivation
inline ScoreActivation stringToScoreActivation(const std::string & activation)
{
  if (activation == "sigmoid" || activation == "SIGMOID") {
    return ScoreActivation::SIGMOID;
  } else if (activation == "softmax" || activation == "SOFTMAX") {
    return ScoreActivation::SOFTMAX;
  } else if (activation == "none" || activation == "NONE") {
    return ScoreActivation::NONE;
  }
  return ScoreActivation::SIGMOID;  // Default
}

}  // namespace deep_object_detection

