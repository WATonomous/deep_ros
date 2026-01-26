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
 * - Enum types for configuration options (BboxFormat, NormalizationType, etc.)
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
  int input_width;  ///< Model input image width in pixels
  int input_height;  ///< Model input image height in pixels
  NormalizationType normalization_type;  ///< Normalization method
  std::vector<float> mean;  ///< Mean values for custom normalization (RGB order)
  std::vector<float> std;  ///< Std values for custom normalization (RGB order)
  ResizeMethod resize_method;  ///< Image resizing method
  int pad_value;  ///< Padding value for letterbox (gray, YOLO standard)
  std::string color_format;  ///< Input color format ("bgr" or "rgb")
};

/**
 * @brief Model metadata configuration
 */
struct ModelMetadata
{
  int num_classes;  ///< Number of detection classes
  std::string class_names_file;  ///< Path to class names file (one per line, optional)
  BboxFormat bbox_format;  ///< Bounding box format used by model
};

/**
 * @brief Output tensor layout configuration
 *
 * Describes the structure of model output tensors for manual layout specification.
 * Only used when auto_detect is false.
 */
struct OutputLayoutConfig
{
  bool auto_detect;  ///< True to auto-detect layout, false to use manual config
  int batch_dim;  ///< Batch dimension index
  int detection_dim;  ///< Detection dimension index
  int feature_dim;  ///< Feature dimension index
  int bbox_start_idx;  ///< Starting index for bbox coordinates in feature dimension
  int bbox_count;  ///< Number of bbox coordinates (always 4)
  int score_idx;  ///< Index for confidence score in feature dimension
  int class_idx;  ///< Index for class ID in feature dimension
};

/**
 * @brief Postprocessing configuration parameters
 */
struct PostprocessingConfig
{
  float score_threshold;  ///< Minimum confidence score (detections below are filtered)
  float nms_iou_threshold;  ///< IoU threshold for Non-Maximum Suppression
  int max_detections;  ///< Maximum number of detections per image (after NMS)
  ScoreActivation score_activation;  ///< Score activation function
  bool enable_nms;  ///< Enable Non-Maximum Suppression
  bool use_multi_output;  ///< True if model has separate outputs for boxes, scores, classes
  int output_boxes_idx;  ///< Output index for bounding boxes (if use_multi_output)
  int output_scores_idx;  ///< Output index for scores (if use_multi_output)
  int output_classes_idx;  ///< Output index for class IDs (if use_multi_output)
  ClassScoreMode class_score_mode;  ///< How class scores are extracted
  int class_score_start_idx;  ///< Start index for class scores (-1 = use all)
  int class_score_count;  ///< Count of class scores (-1 = use all)
  OutputLayoutConfig layout;  ///< Output layout configuration
};

/**
 * @brief Backend configuration parameters
 */
struct BackendConfig
{
  std::string plugin;  ///< Backend plugin name (e.g., "onnxruntime_cpu" or "onnxruntime_gpu")
  std::string execution_provider;  ///< Execution provider for GPU plugins ("cuda" or "tensorrt")
  int device_id;  ///< GPU device ID (for CUDA/TensorRT)
  bool trt_engine_cache_enable;  ///< Enable TensorRT engine caching
  std::string trt_engine_cache_path;  ///< TensorRT engine cache directory
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
  BackendConfig backend;  ///< Backend configuration
  std::string output_detections_topic;  ///< Output topic for detections
  std::vector<std::string> class_names;  ///< Class name strings (loaded from file or empty)
};

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
