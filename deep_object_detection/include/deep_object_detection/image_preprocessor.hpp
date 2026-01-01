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

/**
 * @brief Configurable image preprocessor for object detection models
 *
 * Supports various normalization schemes (ImageNet, scale_0_1, custom, none) and
 * resize methods (letterbox, simple resize, crop, pad).
 */
class ImagePreprocessor
{
public:
  /**
   * @brief Construct an image preprocessor
   *
   * @param config Preprocessing configuration
   */
  explicit ImagePreprocessor(const PreprocessingConfig & config);

  /**
   * @brief Preprocess a single image
   *
   * @param bgr Input BGR image
   * @param meta Output metadata for coordinate transformation
   * @return Preprocessed image (float32, normalized)
   */
  cv::Mat preprocess(const cv::Mat & bgr, ImageMeta & meta) const;

  /**
   * @brief Pack multiple preprocessed images into a batch tensor
   *
   * @param images Vector of preprocessed images
   * @return Packed input tensor data
   */
  const PackedInput & pack(const std::vector<cv::Mat> & images) const;

  /**
   * @brief Get the current preprocessing configuration
   */
  const PreprocessingConfig & config() const { return config_; }

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

