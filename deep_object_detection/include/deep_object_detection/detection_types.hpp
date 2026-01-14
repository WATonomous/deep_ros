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

constexpr size_t RGB_CHANNELS = 3;

enum class Provider
{
  TENSORRT,
  CUDA,
  CPU
};

enum class PostprocessorType
{
  GENERIC
};

enum class BboxFormat
{
  CXCYWH,
  XYXY,
  XYWH
};

enum class NormalizationType
{
  IMAGENET,
  SCALE_0_1,
  CUSTOM,
  NONE
};

enum class ResizeMethod
{
  LETTERBOX,
  RESIZE,
  CROP,
  PAD
};

enum class ScoreActivation
{
  SIGMOID,
  SOFTMAX,
  NONE
};

enum class ClassScoreMode
{
  ALL_CLASSES,
  SINGLE_CONFIDENCE
};

enum class CoordinateSpace
{
  PREPROCESSED,
  ORIGINAL
};

enum class QueueOverflowPolicy
{
  DROP_OLDEST,
  DROP_NEWEST,
  THROW
};

enum class DecodeFailurePolicy
{
  DROP,
  THROW
};

struct ImageMeta
{
  int original_width = 0;
  int original_height = 0;
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
};

struct QueuedImage
{
  cv::Mat bgr;  ///< Decoded BGR image (OpenCV Mat)
  std_msgs::msg::Header header;  ///< ROS message header with timestamp and frame_id
};

struct PackedInput
{
  std::vector<float> data;
  std::vector<size_t> shape;
};

struct PreprocessingConfig
{
  int input_width = 640;
  int input_height = 640;
  NormalizationType normalization_type = NormalizationType::SCALE_0_1;
  std::vector<float> mean = {0.0f, 0.0f, 0.0f};
  std::vector<float> std = {1.0f, 1.0f, 1.0f};
  ResizeMethod resize_method = ResizeMethod::LETTERBOX;
  int pad_value = 114;
  std::string color_format = "bgr";
};

struct ModelMetadata
{
  int num_classes = 80;
  std::string class_names_file;
  BboxFormat bbox_format = BboxFormat::CXCYWH;
};

struct OutputLayoutConfig
{
  bool auto_detect = true;
  int batch_dim = 0;
  int detection_dim = 1;
  int feature_dim = 2;
  int bbox_start_idx = 0;
  int bbox_count = 4;
  int score_idx = 4;
  int class_idx = 5;
};

struct PostprocessingConfig
{
  float score_threshold = 0.25f;
  float nms_iou_threshold = 0.45f;
  int max_detections = 300;
  ScoreActivation score_activation = ScoreActivation::SIGMOID;
  bool enable_nms = true;
  bool use_multi_output = false;
  int output_boxes_idx = 0;
  int output_scores_idx = 1;
  int output_classes_idx = 2;
  ClassScoreMode class_score_mode = ClassScoreMode::ALL_CLASSES;
  int class_score_start_idx = -1;
  int class_score_count = -1;
  CoordinateSpace coordinate_space = CoordinateSpace::PREPROCESSED;
  OutputLayoutConfig layout;
};

struct DetectionParams
{
  std::string model_path;
  ModelMetadata model_metadata;
  PreprocessingConfig preprocessing;
  PostprocessingConfig postprocessing;
  std::string input_qos_reliability{"best_effort"};
  std::string output_detections_topic{"/detections"};
  int max_batch_size{3};
  int queue_size{10};
  QueueOverflowPolicy queue_overflow_policy{QueueOverflowPolicy::DROP_OLDEST};
  DecodeFailurePolicy decode_failure_policy{DecodeFailurePolicy::DROP};
  std::string preferred_provider{"tensorrt"};
  int device_id{0};
  bool warmup_tensor_shapes{true};
  bool enable_trt_engine_cache{false};
  std::string trt_engine_cache_path{"/tmp/deep_ros_ort_trt_cache"};
  std::vector<std::string> class_names;
};

inline PostprocessorType stringToPostprocessorType(const std::string & /*type*/)
{
  return PostprocessorType::GENERIC;
}

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

inline ClassScoreMode stringToClassScoreMode(const std::string & mode)
{
  if (mode == "all_classes" || mode == "ALL_CLASSES") {
    return ClassScoreMode::ALL_CLASSES;
  } else if (mode == "single_confidence" || mode == "SINGLE_CONFIDENCE") {
    return ClassScoreMode::SINGLE_CONFIDENCE;
  }
  return ClassScoreMode::ALL_CLASSES;
}

inline CoordinateSpace stringToCoordinateSpace(const std::string & space)
{
  if (space == "preprocessed" || space == "PREPROCESSED") {
    return CoordinateSpace::PREPROCESSED;
  } else if (space == "original" || space == "ORIGINAL") {
    return CoordinateSpace::ORIGINAL;
  }
  return CoordinateSpace::PREPROCESSED;
}

inline QueueOverflowPolicy stringToQueueOverflowPolicy(const std::string & policy)
{
  if (policy == "drop_oldest" || policy == "DROP_OLDEST") {
    return QueueOverflowPolicy::DROP_OLDEST;
  } else if (policy == "drop_newest" || policy == "DROP_NEWEST") {
    return QueueOverflowPolicy::DROP_NEWEST;
  } else if (policy == "throw" || policy == "THROW") {
    return QueueOverflowPolicy::THROW;
  }
  return QueueOverflowPolicy::DROP_OLDEST;
}

inline DecodeFailurePolicy stringToDecodeFailurePolicy(const std::string & policy)
{
  if (policy == "drop" || policy == "DROP") {
    return DecodeFailurePolicy::DROP;
  } else if (policy == "throw" || policy == "THROW") {
    return DecodeFailurePolicy::THROW;
  }
  return DecodeFailurePolicy::DROP;
}

struct SimpleDetection
{
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float score = 0.0f;
  int32_t class_id = -1;
};

}  // namespace deep_object_detection
