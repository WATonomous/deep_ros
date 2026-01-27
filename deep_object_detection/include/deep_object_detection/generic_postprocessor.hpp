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
 * @file generic_postprocessor.hpp
 * @brief Generic postprocessor for object detection model outputs
 *
 * This header defines the GenericPostprocessor class which:
 * - Uses manually configured output tensor layouts (from YAML config)
 * - Applies score activation and thresholding
 * - Performs Non-Maximum Suppression (NMS)
 * - Transforms coordinates from preprocessed to original image space
 * - Formats detections into ROS Detection2DArray messages
 * - Supports both single-output and multi-output models
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <deep_core/types/tensor.hpp>
#include <std_msgs/msg/header.hpp>

#include "deep_object_detection/detection_types.hpp"

namespace deep_object_detection
{

/**
 * @brief Generic postprocessor for object detection models
 *
 * Handles postprocessing pipeline for various ONNX model output formats:
 * - Manual layout configuration (supports [batch, detections, features], [batch, features, detections], etc.)
 * - Score activation (sigmoid, softmax, or none)
 * - Score thresholding
 * - Non-maximum suppression (NMS)
 * - Coordinate transformation from preprocessed to original image space
 * - Message formatting for ROS Detection2DArray
 *
 * Supports both single-output models (boxes + scores + classes in one tensor)
 * and multi-output models (separate tensors for boxes, scores, classes).
 * Layout must be manually configured in the YAML config file.
 */
class GenericPostprocessor
{
public:
  /**
   * @brief Output tensor layout configuration
   *
   * Describes the structure of model output tensors:
   * - Dimension indices for batch, detections, and features
   * - Feature indices for bbox coordinates, scores, and class IDs
   * - Support for transposed layouts and multi-output models
   */
  struct OutputLayout
  {
    std::vector<size_t> shape;  ///< Output tensor shape
    size_t batch_dim = 0;  ///< Batch dimension index
    size_t detection_dim = 1;  ///< Detection dimension index
    size_t feature_dim = 2;  ///< Feature dimension index
    size_t bbox_start_idx = 0;  ///< Starting index for bbox coordinates in feature dimension
    size_t bbox_count = 4;  ///< Number of bbox coordinates (always 4)
    size_t score_idx = 4;  ///< Index for confidence score in feature dimension
    size_t class_idx = 5;  ///< Index for class ID in feature dimension
    bool has_separate_class_output = false;  ///< True if class IDs are in separate output tensor
    size_t class_output_idx = 0;  ///< Output index for separate class tensor (if applicable)
  };

  /**
   * @brief Construct the postprocessor
   * @param config Postprocessing configuration (thresholds, NMS, activation)
   * @param layout Output tensor layout configuration
   * @param bbox_format Bounding box format (cxcywh, xyxy, or xywh)
   * @param num_classes Number of detection classes
   * @param class_names Vector of class name strings (empty if using class IDs)
   * @param use_letterbox True if letterbox resize was used (affects coordinate transformation)
   */
  GenericPostprocessor(
    const PostprocessingConfig & config,
    const OutputLayout & layout,
    const std::string & bbox_format,
    int num_classes,
    const std::vector<std::string> & class_names,
    bool use_letterbox);

  /**
   * @brief Configure output layout from manual configuration
   * @param output_shape Model output shape (optional, for validation/logging)
   * @param layout_config Layout configuration from parameters
   * @return Configured OutputLayout
   *
   * Creates OutputLayout from manual configuration parameters.
   * All layout parameters must be specified in the config file.
   */
  static OutputLayout configureLayout(
    const std::vector<size_t> & output_shape, const OutputLayoutConfig & layout_config);

  /**
   * @brief Decode model output tensor to detections
   * @param output Model output tensor (single output)
   * @param metas Image metadata for coordinate transformation
   * @return Vector of detections per image in batch
   *
   * Extracts bounding boxes, scores, and class IDs from output tensor.
   * Applies score thresholding, NMS, and coordinate transformation.
   * Returns one vector of detections per image in the batch.
   */
  std::vector<std::vector<SimpleDetection>> decode(
    const deep_ros::Tensor & output, const std::vector<ImageMeta> & metas) const;

  /**
   * @brief Fill ROS Detection2DArray message with detections
   * @param header ROS message header (timestamp and frame_id)
   * @param detections Vector of detections for one image
   * @param meta Image metadata
   * @param out_msg Output message to fill
   *
   * Converts SimpleDetection objects to ROS Detection2D messages
   * and populates the Detection2DArray message.
   */
  void fillDetectionMessage(
    const std_msgs::msg::Header & header,
    const std::vector<SimpleDetection> & detections,
    const ImageMeta & meta,
    Detection2DArrayMsg & out_msg) const;

protected:
  /**
   * @brief Transform detection coordinates from preprocessed to original image space
   * @param det Detection to transform (modified in-place)
   * @param meta Image metadata (scale, padding)
   * @param use_letterbox True if letterbox resize was used
   *
   * Transforms bounding box coordinates from preprocessed image space
   * (model input size) to original image space using scale and padding
   * information from preprocessing.
   */
  void adjustToOriginal(SimpleDetection & det, const ImageMeta & meta, bool use_letterbox) const;

  /**
   * @brief Apply Non-Maximum Suppression to detections
   * @param dets Vector of detections (modified in-place)
   * @param iou_threshold IoU threshold for NMS
   * @return Filtered detections after NMS
   *
   * Removes overlapping detections with lower scores. Detections are
   * sorted by score (descending) and suppressed if IoU > threshold.
   */
  std::vector<SimpleDetection> applyNms(std::vector<SimpleDetection> dets, float iou_threshold) const;

  /**
   * @brief Calculate Intersection over Union (IoU) between two detections
   * @param a First detection
   * @param b Second detection
   * @return IoU value [0, 1]
   */
  static float iou(const SimpleDetection & a, const SimpleDetection & b);

  /**
   * @brief Get class label string for a class ID
   * @param class_id Class ID
   * @param class_names Vector of class name strings
   * @return Class label string (name if available, otherwise ID as string)
   */
  std::string classLabel(int class_id, const std::vector<std::string> & class_names) const;

  /**
   * @brief Apply score activation function
   * @param raw_score Raw score from model
   * @return Activated score
   *
   * Applies activation according to config:
   * - sigmoid: 1 / (1 + exp(-x))
   * - softmax: exp(x) / sum(exp(x)) (not applicable here, returns raw)
   * - none: return raw score
   */
  float applyActivation(float raw_score) const;

private:
  /**
   * @brief Extract a single value from output tensor
   * @param data Pointer to tensor data (flattened array)
   * @param batch_idx Batch index
   * @param detection_idx Detection index
   * @param feature_idx Feature index
   * @param shape Tensor shape for index calculation
   * @return Extracted float value
   *
   * Calculates linear index from multi-dimensional indices and extracts value.
   * Handles both standard and transposed layouts.
   */
  float extractValue(
    const float * data,
    size_t batch_idx,
    size_t detection_idx,
    size_t feature_idx,
    const std::vector<size_t> & shape) const;

  /**
   * @brief Convert bbox data from tensor to SimpleDetection
   * @param bbox_data Pointer to bbox coordinates in tensor
   * @param batch_idx Batch index
   * @param detection_idx Detection index
   * @param shape Tensor shape
   * @param det Output detection (bbox coordinates filled)
   *
   * Extracts 4 bbox coordinates and converts from model format (cxcywh, xyxy, or xywh)
   * to SimpleDetection format (x, y, width, height in original image space).
   */
  void convertBbox(
    const float * bbox_data,
    size_t batch_idx,
    size_t detection_idx,
    const std::vector<size_t> & shape,
    SimpleDetection & det) const;

  /// Postprocessing configuration (thresholds, NMS, activation)
  PostprocessingConfig config_;
  /// Output tensor layout configuration
  OutputLayout layout_;
  /// Bounding box format (cxcywh, xyxy, or xywh)
  std::string bbox_format_;
  /// Number of detection classes
  int num_classes_;
  /// Class name strings (empty if using class IDs)
  std::vector<std::string> class_names_;
  /// True if letterbox resize was used (affects coordinate transformation)
  bool use_letterbox_;
};

}  // namespace deep_object_detection
