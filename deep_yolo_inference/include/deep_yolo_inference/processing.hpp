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

#include <deep_core/types/tensor.hpp>
#include <opencv2/core/mat.hpp>
#include <std_msgs/msg/header.hpp>

#include "deep_yolo_inference/detection_msg_alias.hpp"
#include "deep_yolo_inference/yolo_types.hpp"

namespace deep_yolo_inference
{

class ImagePreprocessor
{
public:
  explicit ImagePreprocessor(const YoloParams & params);

  cv::Mat preprocess(const cv::Mat & bgr, ImageMeta & meta) const;
  const PackedInput & pack(const std::vector<cv::Mat> & images) const;

private:
  const YoloParams & params_;
  mutable PackedInput packed_input_cache_;
};

class Postprocessor
{
public:
  Postprocessor(const YoloParams & params, const std::vector<std::string> & class_names);

  std::vector<std::vector<SimpleDetection>> decode(
    const deep_ros::Tensor & output, const std::vector<ImageMeta> & metas) const;

  void fillDetectionMessage(
    const std_msgs::msg::Header & header,
    const std::vector<SimpleDetection> & detections,
    const ImageMeta & meta,
    Detection2DArrayMsg & out_msg) const;

private:
  void adjustToOriginal(SimpleDetection & det, const ImageMeta & meta) const;
  std::vector<SimpleDetection> applyNms(std::vector<SimpleDetection> dets) const;
  static float iou(const SimpleDetection & a, const SimpleDetection & b);
  std::string classLabel(int class_id) const;

  const YoloParams & params_;
  const std::vector<std::string> & class_names_;
};

}  // namespace deep_yolo_inference
