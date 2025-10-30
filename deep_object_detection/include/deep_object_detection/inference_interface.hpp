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

#include <opencv2/opencv.hpp>

namespace deep_object_detection
{

struct Detection
{
  float x, y, width, height;  // Bounding box
  float confidence;  // Detection confidence
  int class_id;  // Class ID
  std::string class_name;  // Class name
};

enum class InferenceBackend
{
  AUTO,  // Automatically select best available backend
  ORT_BACKEND  // Use zero-copy ORT backend (recommended)
};

struct InferenceConfig
{
  std::string model_path;
  std::vector<std::string> class_names;
  int input_width = 640;
  int input_height = 640;
  float confidence_threshold = 0.5f;
  float nms_threshold = 0.4f;
  int max_batch_size = 8;
  bool use_gpu = true;
  std::string input_blob_name = "images";
  std::string output_blob_name = "output0";
  InferenceBackend backend = InferenceBackend::AUTO;
};

/**
 * @brief Abstract interface for object detection inference engines
 */
class InferenceInterface
{
public:
  virtual ~InferenceInterface() = default;

  /**
     * @brief Initialize the inference engine
     * @return true if initialization successful, false otherwise
     */
  virtual bool initialize() = 0;

  /**
     * @brief Perform batch inference on multiple images
     * @param images Vector of input images
     * @return Vector of detection results for each image
     */
  virtual std::vector<std::vector<Detection>> inferBatch(const std::vector<cv::Mat> & images) = 0;

  /**
     * @brief Perform inference on a single image (convenience method)
     * @param image Input image
     * @return Vector of detections for the image
     */
  virtual std::vector<Detection> infer(const cv::Mat & image) = 0;

  /**
     * @brief Get input width expected by the model
     * @return Input width in pixels
     */
  virtual int getInputWidth() const = 0;

  /**
     * @brief Get input height expected by the model
     * @return Input height in pixels
     */
  virtual int getInputHeight() const = 0;

  /**
     * @brief Get maximum batch size supported
     * @return Maximum batch size
     */
  virtual int getMaxBatchSize() const = 0;

  /**
     * @brief Check if the inference engine is initialized
     * @return true if initialized, false otherwise
     */
  virtual bool isInitialized() const = 0;
};

/**
 * @brief Factory function to create inference engines
 * @param config Inference configuration
 * @return Unique pointer to inference engine instance
 */
std::unique_ptr<InferenceInterface> createInferenceEngine(const InferenceConfig & config);

}  // namespace deep_object_detection
