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

#include "deep_object_detection/detection_types.hpp"

namespace deep_object_detection
{

class ImagePreprocessor
{
public:
  explicit ImagePreprocessor(const PreprocessingConfig & config);
  cv::Mat preprocess(const cv::Mat & bgr, ImageMeta & meta) const;
  const PackedInput & pack(const std::vector<cv::Mat> & images) const;

  const PreprocessingConfig & config() const
  {
    return config_;
  }

private:
  cv::Mat applyLetterbox(const cv::Mat & bgr, ImageMeta & meta) const;
  cv::Mat applyResize(const cv::Mat & bgr, ImageMeta & meta) const;
  cv::Mat applyCrop(const cv::Mat & bgr, ImageMeta & meta) const;
  cv::Mat applyPad(const cv::Mat & bgr, ImageMeta & meta) const;
  void applyNormalization(cv::Mat & image) const;
  void applyColorConversion(cv::Mat & image) const;

  PreprocessingConfig config_;
  mutable PackedInput packed_input_cache_;
};

}  // namespace deep_object_detection
