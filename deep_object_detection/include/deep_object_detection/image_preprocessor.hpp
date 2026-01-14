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
 * @file image_preprocessor.hpp
 * @brief Image preprocessing for object detection models
 *
 * This header defines the ImagePreprocessor class which:
 * - Resizes images (letterbox, resize, crop, or pad)
 * - Converts color formats (BGR to RGB)
 * - Applies normalization (scale_0_1, imagenet, custom, or none)
 * - Batches multiple images into a single tensor
 * - Tracks metadata for coordinate transformation
 */

#pragma once

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "deep_object_detection/detection_types.hpp"

namespace deep_object_detection
{

/**
 * @brief Preprocesses images for model input
 *
 * Handles image preprocessing pipeline:
 * - Resizing (letterbox, resize, crop, or pad)
 * - Color format conversion (BGR to RGB)
 * - Normalization (scale_0_1, imagenet, custom, or none)
 * - Batching multiple images into a single tensor
 *
 * The preprocessor maintains aspect ratio when using letterbox resize
 * and tracks metadata (scale, padding) for coordinate transformation
 * in postprocessing.
 */
class ImagePreprocessor
{
public:
  /**
   * @brief Construct the preprocessor with configuration
   * @param config Preprocessing configuration (resize method, normalization, etc.)
   */
  explicit ImagePreprocessor(const PreprocessingConfig & config);

  /**
   * @brief Preprocess a single BGR image
   * @param bgr Input BGR image (OpenCV Mat)
   * @param meta Output metadata (original size, scale, padding) for coordinate transformation
   * @return Preprocessed image ready for model input
   *
   * Applies resize, color conversion, and normalization according to config.
   * Updates meta with transformation parameters needed for postprocessing.
   */
  cv::Mat preprocess(const cv::Mat & bgr, ImageMeta & meta) const;

  /**
   * @brief Pack a batch of preprocessed images into a single tensor
   * @param images Vector of preprocessed images (same size, normalized)
   * @return Reference to PackedInput containing flattened float array and shape
   *
   * Converts batch of OpenCV Mats to a single flattened float array
   * in NCHW format (batch, channels, height, width). Uses internal cache
   * to avoid reallocation. Thread-safe for read-only access.
   */
  const PackedInput & pack(const std::vector<cv::Mat> & images) const;

  /**
   * @brief Get preprocessing configuration
   * @return Reference to preprocessing configuration
   */
  const PreprocessingConfig & config() const
  {
    return config_;
  }

private:
  /**
   * @brief Apply letterbox resize (maintain aspect ratio, pad with gray)
   * @param bgr Input BGR image
   * @param meta Output metadata (scale and padding info)
   * @return Resized image with letterbox padding
   *
   * Maintains aspect ratio by scaling to fit within input dimensions,
   * then pads with gray (114) to reach exact input size. Updates meta
   * with scale factors and padding offsets.
   */
  cv::Mat applyLetterbox(const cv::Mat & bgr, ImageMeta & meta) const;

  /**
   * @brief Apply simple resize (stretch to input size)
   * @param bgr Input BGR image
   * @param meta Output metadata (scale info)
   * @return Resized image (may distort aspect ratio)
   *
   * Stretches image to exact input dimensions. Updates meta with scale factors.
   */
  cv::Mat applyResize(const cv::Mat & bgr, ImageMeta & meta) const;

  /**
   * @brief Apply center crop to input size
   * @param bgr Input BGR image
   * @param meta Output metadata (scale info)
   * @return Cropped image
   *
   * Crops center region to input dimensions. Updates meta with scale factors.
   */
  cv::Mat applyCrop(const cv::Mat & bgr, ImageMeta & meta) const;

  /**
   * @brief Apply padding to input size
   * @param bgr Input BGR image
   * @param meta Output metadata (padding info)
   * @return Padded image
   *
   * Pads image with gray (114) to reach input dimensions. Updates meta with padding offsets.
   */
  cv::Mat applyPad(const cv::Mat & bgr, ImageMeta & meta) const;

  /**
   * @brief Apply normalization to image
   * @param image Image to normalize (modified in-place)
   *
   * Applies normalization according to config:
   * - scale_0_1: divide by 255.0
   * - imagenet: subtract mean [0.485, 0.456, 0.406], divide by std [0.229, 0.224, 0.225]
   * - custom: use config mean and std
   * - none: no normalization
   */
  void applyNormalization(cv::Mat & image) const;

  /**
   * @brief Convert BGR to RGB if needed
   * @param image Image to convert (modified in-place)
   *
   * Converts BGR to RGB if color_format is "rgb" in config.
   * No-op if color_format is "bgr".
   */
  void applyColorConversion(cv::Mat & image) const;

  /// Preprocessing configuration
  PreprocessingConfig config_;
  /// Cached packed input (reused to avoid reallocation)
  mutable PackedInput packed_input_cache_;
};

}  // namespace deep_object_detection
