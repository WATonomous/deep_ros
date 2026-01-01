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

#include <memory>
#include <string>
#include <vector>

#include <deep_core/types/tensor.hpp>
#include <std_msgs/msg/header.hpp>

#include "deep_object_detection/detection_msg_alias.hpp"
#include "deep_object_detection/detection_types.hpp"

namespace deep_object_detection
{

/**
 * @brief Generic configurable postprocessor for any ONNX detection model
 *
 * This postprocessor can handle any output format by being configured via parameters
 * that describe the output tensor structure. It supports:
 * - Flexible tensor layouts (any dimension order)
 * - Configurable coordinate formats (cxcywh, xyxy, xywh)
 * - Configurable score/class extraction
 * - Automatic format detection from model metadata
 */
class GenericPostprocessor
{
public:
  /**
   * @brief Output tensor layout configuration
   */
  struct OutputLayout
  {
    std::vector<size_t> shape;           // Expected output shape [batch, ...]
    size_t batch_dim = 0;               // Which dimension is batch (usually 0)
    size_t detection_dim = 1;           // Which dimension contains detections/queries
    size_t feature_dim = 2;             // Which dimension contains features (bbox+score+class)
    
    // Feature indices (within feature_dim)
    size_t bbox_start_idx = 0;          // Start index of bbox coordinates
    size_t bbox_count = 4;              // Number of bbox values (usually 4)
    size_t score_idx = 4;                // Index of score value
    size_t class_idx = 5;                // Index of class_id value (or -1 if separate)
    
    bool has_separate_class_output = false;  // If class is in separate output tensor
    size_t class_output_idx = 0;        // Which output tensor contains classes (if separate)
    
    // Layout detection
    bool auto_detect = true;            // Auto-detect layout from shape
  };

  /**
   * @brief Construct a generic postprocessor
   *
   * @param config Postprocessing configuration
   * @param layout Output layout configuration
   * @param bbox_format Bounding box format
   * @param num_classes Number of classes
   * @param class_names Vector of class names
   * @param use_letterbox Whether letterbox preprocessing was used
   */
  GenericPostprocessor(
    const PostprocessingConfig & config,
    const OutputLayout & layout,
    BboxFormat bbox_format,
    int num_classes,
    const std::vector<std::string> & class_names,
    bool use_letterbox);

  /**
   * @brief Auto-detect output layout from tensor shape
   *
   * @param output_shape Output tensor shape
   * @return Detected layout configuration
   */
  static OutputLayout detectLayout(const std::vector<size_t> & output_shape);

  std::vector<std::vector<SimpleDetection>> decode(
    const deep_ros::Tensor & output,
    const std::vector<ImageMeta> & metas) const;

  std::string getFormatName() const { return "generic"; }

  void fillDetectionMessage(
    const std_msgs::msg::Header & header,
    const std::vector<SimpleDetection> & detections,
    const ImageMeta & meta,
    Detection2DArrayMsg & out_msg) const;

protected:
  /**
   * @brief Adjust detection coordinates to original image space
   */
  void adjustToOriginal(SimpleDetection & det, const ImageMeta & meta, bool use_letterbox) const;

  /**
   * @brief Apply Non-Maximum Suppression to detections
   */
  std::vector<SimpleDetection> applyNms(
    std::vector<SimpleDetection> dets,
    float iou_threshold) const;

  /**
   * @brief Compute Intersection over Union between two detections
   */
  static float iou(const SimpleDetection & a, const SimpleDetection & b);

  /**
   * @brief Get class label for a given class ID
   */
  std::string classLabel(int class_id, const std::vector<std::string> & class_names) const;

private:
  /**
   * @brief Extract a value from the output tensor based on layout
   */
  float extractValue(
    const float * data,
    size_t batch_idx,
    size_t detection_idx,
    size_t feature_idx,
    const std::vector<size_t> & shape) const;

  /**
   * @brief Convert bbox coordinates based on format
   */
  void convertBbox(
    const float * bbox_data,
    size_t batch_idx,
    size_t detection_idx,
    const std::vector<size_t> & shape,
    SimpleDetection & det) const;

  PostprocessingConfig config_;
  OutputLayout layout_;
  BboxFormat bbox_format_;
  int num_classes_;
  std::vector<std::string> class_names_;
  bool use_letterbox_;
};

}  // namespace deep_object_detection

