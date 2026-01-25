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

/**
 * @file detection_types.hpp
 * @brief Type definitions and enums for deep object detection
 *
 * This header defines:
 * - Enum types for configuration options (Provider, BboxFormat, NormalizationType, etc.)
 * - Configuration structures (PreprocessingConfig, PostprocessingConfig, DetectionParams)
 * - Data structures (ImageMeta, PackedInput, SimpleDetection)
 * - Helper functions for string-to-enum conversion
 * - ROS message type aliases (Detection2DMsg, Detection2DArrayMsg)
 */

#pragma once

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <std_msgs/msg/header.hpp>

#if __has_include(<deep_msgs/msg/detection2_d_array.hpp>)
  #include <deep_msgs/msg/detection2_d.hpp>
  #include <deep_msgs/msg/detection2_d_array.hpp>

namespace deep_object_detection
{
using Detection2DMsg = deep_msgs::msg::Detection2D;
using Detection2DArrayMsg = deep_msgs::msg::Detection2DArray;
}  // namespace deep_object_detection
#else
  #include <vision_msgs/msg/detection2_d.hpp>
  #include <vision_msgs/msg/detection2_d_array.hpp>
  #include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

namespace deep_object_detection
{
using Detection2DMsg = vision_msgs::msg::Detection2D;
using Detection2DArrayMsg = vision_msgs::msg::Detection2DArray;
}  // namespace deep_object_detection
#endif

namespace deep_object_detection
{

/// Number of RGB color channels
constexpr size_t RGB_CHANNELS = 3;

/**
 * @brief Execution provider for ONNX Runtime inference
 */
enum class Provider
{
  TENSORRT,  ///< TensorRT execution provider (requires CUDA and TensorRT)
  CUDA,  ///< CUDA execution provider (requires CUDA)
  CPU  ///< CPU execution provider (always available)
};

/**
 * @brief Bounding box coordinate format
 */
enum class BboxFormat
{
  CXCYWH,  ///< Center x, center y, width, height (YOLO format)
  XYXY,  ///< Top-left x, top-left y, bottom-right x, bottom-right y
  XYWH  ///< Top-left x, top-left y, width, height
};

/**
 * @brief Image normalization method
 */
enum class NormalizationType
{
  IMAGENET,  ///< ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  SCALE_0_1,  ///< Scale to [0, 1] range (divide by 255.0)
  CUSTOM,  ///< Custom mean and std values
  NONE  ///< No normalization
};

/**
 * @brief Image resizing method
 */
enum class ResizeMethod
{
  LETTERBOX,  ///< Maintain aspect ratio, pad with gray (114) to fit input size
  RESIZE,  ///< Stretch to input size (may distort aspect ratio)
  CROP,  ///< Center crop to input size
  PAD  ///< Pad to input size
};

/**
 * @brief Score activation function for raw model outputs
 */
enum class ScoreActivation
{
  SIGMOID,  ///< Sigmoid activation: 1 / (1 + exp(-x))
  SOFTMAX,  ///< Softmax activation (not fully implemented, returns raw)
  NONE  ///< No activation (use raw scores)
};

/**
 * @brief How class scores are extracted from model output
 */
enum class ClassScoreMode
{
  ALL_CLASSES,  ///< Extract class scores from all class logits (multi-class)
  SINGLE_CONFIDENCE  ///< Use a single confidence score (single-class models)
};

/**
 * @brief Coordinate space for bounding boxes
 */
enum class CoordinateSpace
{
  PREPROCESSED  ///< Coordinates in preprocessed image space (model input size)
};

/**
 * @brief Image metadata for coordinate transformation
 *
 * Stores information about preprocessing transformations needed to
 * transform bounding box coordinates from preprocessed image space
 * back to original image space.
 */
struct ImageMeta
{
  int original_width = 0;  ///< Original image width before preprocessing
  int original_height = 0;  ///< Original image height before preprocessing
  float scale_x = 1.0f;  ///< Horizontal scale factor (original_width / input_width)
  float scale_y = 1.0f;  ///< Vertical scale factor (original_height / input_height)
  float pad_x = 0.0f;  ///< Horizontal padding offset (for letterbox)
  float pad_y = 0.0f;  ///< Vertical padding offset (for letterbox)
};

/**
 * @brief Packed input tensor for model inference
 *
 * Flattened float array in NCHW format (batch, channels, height, width)
 * ready for model input. Shape vector describes tensor dimensions.
 */
struct PackedInput
{
  std::vector<float> data;  ///< Flattened float array (NCHW format)
  std::vector<size_t> shape;  ///< Tensor shape [batch, channels, height, width]
};

/**
 * @brief Preprocessing configuration parameters
 */
struct PreprocessingConfig
{
  int input_width = 640;  ///< Model input image width in pixels
  int input_height = 640;  ///< Model input image height in pixels
  NormalizationType normalization_type = NormalizationType::SCALE_0_1;  ///< Normalization method
  std::vector<float> mean = {0.0f, 0.0f, 0.0f};  ///< Mean values for custom normalization (RGB order)
  std::vector<float> std = {1.0f, 1.0f, 1.0f};  ///< Std values for custom normalization (RGB order)
  ResizeMethod resize_method = ResizeMethod::LETTERBOX;  ///< Image resizing method
  int pad_value = 114;  ///< Padding value for letterbox (gray, YOLO standard)
  std::string color_format = "bgr";  ///< Input color format ("bgr" or "rgb")
};

/**
 * @brief Model metadata configuration
 */
struct ModelMetadata
{
  int num_classes = 80;  ///< Number of detection classes
  std::string class_names_file;  ///< Path to class names file (one per line, optional)
  BboxFormat bbox_format = BboxFormat::CXCYWH;  ///< Bounding box format used by model
};

/**
 * @brief Output tensor layout configuration
 *
 * Describes the structure of model output tensors for manual layout specification.
 * Only used when auto_detect is false.
 */
struct OutputLayoutConfig
{
  bool auto_detect = true;  ///< True to auto-detect layout, false to use manual config
  int batch_dim = 0;  ///< Batch dimension index
  int detection_dim = 1;  ///< Detection dimension index
  int feature_dim = 2;  ///< Feature dimension index
  int bbox_start_idx = 0;  ///< Starting index for bbox coordinates in feature dimension
  int bbox_count = 4;  ///< Number of bbox coordinates (always 4)
  int score_idx = 4;  ///< Index for confidence score in feature dimension
  int class_idx = 5;  ///< Index for class ID in feature dimension
};

/**
 * @brief Postprocessing configuration parameters
 */
struct PostprocessingConfig
{
  float score_threshold = 0.25f;  ///< Minimum confidence score (detections below are filtered)
  float nms_iou_threshold = 0.45f;  ///< IoU threshold for Non-Maximum Suppression
  int max_detections = 300;  ///< Maximum number of detections per image (after NMS)
  ScoreActivation score_activation = ScoreActivation::SIGMOID;  ///< Score activation function
  bool enable_nms = true;  ///< Enable Non-Maximum Suppression (always true, not configurable)
  bool use_multi_output = false;  ///< True if model has separate outputs for boxes, scores, classes
  int output_boxes_idx = 0;  ///< Output index for bounding boxes (if use_multi_output)
  int output_scores_idx = 1;  ///< Output index for scores (if use_multi_output)
  int output_classes_idx = 2;  ///< Output index for class IDs (if use_multi_output)
  ClassScoreMode class_score_mode = ClassScoreMode::ALL_CLASSES;  ///< How class scores are extracted
  int class_score_start_idx = -1;  ///< Start index for class scores (-1 = use all)
  int class_score_count = -1;  ///< Count of class scores (-1 = use all)
  CoordinateSpace coordinate_space = CoordinateSpace::PREPROCESSED;  ///< Coordinate space (always PREPROCESSED)
  OutputLayoutConfig layout;  ///< Output layout configuration
};

/**
 * @brief Complete detection node configuration parameters
 *
 * Aggregates all configuration for model, preprocessing, postprocessing,
 * execution provider, batching, and topics.
 */
struct DetectionParams
{
  std::string model_path;  ///< Absolute path to ONNX model file
  ModelMetadata model_metadata;  ///< Model metadata (classes, bbox format)
  PreprocessingConfig preprocessing;  ///< Preprocessing configuration
  PostprocessingConfig postprocessing;  ///< Postprocessing configuration
  std::string output_detections_topic{"/detections"};  ///< Output topic for detections
  std::string preferred_provider{"tensorrt"};  ///< Preferred execution provider ("tensorrt", "cuda", or "cpu")
  int device_id{0};  ///< GPU device ID (for CUDA/TensorRT)
  bool warmup_tensor_shapes{true};  ///< Warmup tensor shapes (always true, not configurable)
  bool enable_trt_engine_cache{false};  ///< Enable TensorRT engine caching
  std::string trt_engine_cache_path{"/tmp/deep_ros_ort_trt_cache"};  ///< TensorRT engine cache directory
  std::vector<std::string> class_names;  ///< Class name strings (loaded from file or empty)
};

/**
 * @brief Convert string to BboxFormat enum
 * @param format Format string ("cxcywh", "xyxy", or "xywh", case-insensitive)
 * @return BboxFormat enum (defaults to CXCYWH if unknown)
 */
inline BboxFormat stringToBboxFormat(const std::string & format)
{
  if (format == "cxcywh" || format == "CXCYWH") {
    return BboxFormat::CXCYWH;
  } else if (format == "xyxy" || format == "XYXY") {
    return BboxFormat::XYXY;
  } else if (format == "xywh" || format == "XYWH") {
    return BboxFormat::XYWH;
  }
  return BboxFormat::CXCYWH;
}

/**
 * @brief Convert string to NormalizationType enum
 * @param type Normalization type string ("imagenet", "scale_0_1", "yolo", "custom", or "none", case-insensitive)
 * @return NormalizationType enum (defaults to SCALE_0_1 if unknown)
 */
inline NormalizationType stringToNormalizationType(const std::string & type)
{
  if (type == "imagenet" || type == "IMAGENET") {
    return NormalizationType::IMAGENET;
  } else if (type == "scale_0_1" || type == "SCALE_0_1" || type == "yolo" || type == "YOLO") {
    return NormalizationType::SCALE_0_1;
  } else if (type == "custom" || type == "CUSTOM") {
    return NormalizationType::CUSTOM;
  } else if (type == "none" || type == "NONE") {
    return NormalizationType::NONE;
  }
  return NormalizationType::SCALE_0_1;
}

/**
 * @brief Convert string to ResizeMethod enum
 * @param method Resize method string ("letterbox", "resize", "crop", or "pad", case-insensitive)
 * @return ResizeMethod enum (defaults to LETTERBOX if unknown)
 */
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
  return ResizeMethod::LETTERBOX;
}

/**
 * @brief Convert string to ScoreActivation enum
 * @param activation Activation string ("sigmoid", "softmax", or "none", case-insensitive)
 * @return ScoreActivation enum (defaults to SIGMOID if unknown)
 */
inline ScoreActivation stringToScoreActivation(const std::string & activation)
{
  if (activation == "sigmoid" || activation == "SIGMOID") {
    return ScoreActivation::SIGMOID;
  } else if (activation == "softmax" || activation == "SOFTMAX") {
    return ScoreActivation::SOFTMAX;
  } else if (activation == "none" || activation == "NONE") {
    return ScoreActivation::NONE;
  }
  return ScoreActivation::SIGMOID;
}

/**
 * @brief Convert string to ClassScoreMode enum
 * @param mode Mode string ("all_classes" or "single_confidence", case-insensitive)
 * @return ClassScoreMode enum (defaults to ALL_CLASSES if unknown)
 */
inline ClassScoreMode stringToClassScoreMode(const std::string & mode)
{
  if (mode == "all_classes" || mode == "ALL_CLASSES") {
    return ClassScoreMode::ALL_CLASSES;
  } else if (mode == "single_confidence" || mode == "SINGLE_CONFIDENCE") {
    return ClassScoreMode::SINGLE_CONFIDENCE;
  }
  return ClassScoreMode::ALL_CLASSES;
}

/**
 * @brief Convert string to CoordinateSpace enum
 * @param space Space string ("preprocessed" or "original", case-insensitive)
 * @return CoordinateSpace enum (defaults to PREPROCESSED if unknown)
 */
inline CoordinateSpace stringToCoordinateSpace(const std::string & space)
{
  if (space == "preprocessed" || space == "PREPROCESSED") {
    return CoordinateSpace::PREPROCESSED;
  }
  return CoordinateSpace::PREPROCESSED;
}

/**
 * @brief Simple detection structure (internal representation)
 *
 * Stores a single detection with bounding box, score, and class ID.
 * Coordinates are in original image space (x, y, width, height).
 */
struct SimpleDetection
{
  float x = 0.0f;  ///< Top-left x coordinate in original image space
  float y = 0.0f;  ///< Top-left y coordinate in original image space
  float width = 0.0f;  ///< Bounding box width in original image space
  float height = 0.0f;  ///< Bounding box height in original image space
  float score = 0.0f;  ///< Confidence score [0, 1]
  int32_t class_id = -1;  ///< Class ID (-1 if unknown)
};

}  // namespace deep_object_detection
