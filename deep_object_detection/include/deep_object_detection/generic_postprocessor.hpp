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

#include "deep_object_detection/detection_types.hpp"

namespace deep_object_detection
{

class GenericPostprocessor
{
public:
  struct OutputLayout
  {
    std::vector<size_t> shape;
    size_t batch_dim = 0;
    size_t detection_dim = 1;
    size_t feature_dim = 2;
    size_t bbox_start_idx = 0;
    size_t bbox_count = 4;
    size_t score_idx = 4;
    size_t class_idx = 5;
    bool has_separate_class_output = false;
    size_t class_output_idx = 0;
    bool auto_detect = true;
  };

  GenericPostprocessor(
    const PostprocessingConfig & config,
    const OutputLayout & layout,
    BboxFormat bbox_format,
    int num_classes,
    const std::vector<std::string> & class_names,
    bool use_letterbox);

  static OutputLayout detectLayout(const std::vector<size_t> & output_shape);

  /**
   * @brief Auto-configure output layout based on config and optional output shape
   * @param output_shape Model output shape (can be empty for deferred detection)
   * @param layout_config Layout configuration from parameters
   * @return Configured OutputLayout
   *
   * Handles both manual and auto-detection modes. If auto_detect is true and output_shape
   * is available, automatically detects layout. Otherwise uses manual config or defers detection.
   */
  static OutputLayout autoConfigure(const std::vector<size_t> & output_shape, const OutputLayoutConfig & layout_config);

  std::vector<std::vector<SimpleDetection>> decode(
    const deep_ros::Tensor & output, const std::vector<ImageMeta> & metas) const;

  std::vector<std::vector<SimpleDetection>> decodeMultiOutput(
    const std::vector<deep_ros::Tensor> & outputs, const std::vector<ImageMeta> & metas) const;

  std::string getFormatName() const
  {
    return "generic";
  }

  void fillDetectionMessage(
    const std_msgs::msg::Header & header,
    const std::vector<SimpleDetection> & detections,
    const ImageMeta & meta,
    Detection2DArrayMsg & out_msg) const;

protected:
  void adjustToOriginal(SimpleDetection & det, const ImageMeta & meta, bool use_letterbox) const;
  std::vector<SimpleDetection> applyNms(std::vector<SimpleDetection> dets, float iou_threshold) const;
  static float iou(const SimpleDetection & a, const SimpleDetection & b);
  std::string classLabel(int class_id, const std::vector<std::string> & class_names) const;
  float applyActivation(float raw_score) const;

private:
  float extractValue(
    const float * data,
    size_t batch_idx,
    size_t detection_idx,
    size_t feature_idx,
    const std::vector<size_t> & shape) const;

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
