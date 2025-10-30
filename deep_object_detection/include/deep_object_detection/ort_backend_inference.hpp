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
#include <mutex>
#include <string>
#include <vector>

#include <deep_core/types/tensor.hpp>
#include <opencv2/opencv.hpp>

#include "deep_object_detection/inference_interface.hpp"
#include "deep_ort_backend_plugin/ort_backend_executor.hpp"

namespace deep_object_detection
{

/**
 * @brief ORT Backend inference adapter for object detection
 *
 * This class adapts the deep_ort_backend::OrtBackendExecutor to work with
 * the object detection inference interface, providing zero-copy inference
 * capabilities while maintaining compatibility with OpenCV images.
 */
class OrtBackendInference : public InferenceInterface
{
public:
  explicit OrtBackendInference(const InferenceConfig & config);
  ~OrtBackendInference() override;

  // InferenceInterface implementation
  bool initialize() override;
  std::vector<Detection> infer(const cv::Mat & image) override;
  std::vector<std::vector<Detection>> inferBatch(const std::vector<cv::Mat> & images) override;

  int getInputWidth() const override
  {
    return config_.input_width;
  }

  int getInputHeight() const override
  {
    return config_.input_height;
  }

  int getMaxBatchSize() const override
  {
    return config_.max_batch_size;
  }

  bool isInitialized() const override
  {
    return initialized_;
  }

private:
  // Configuration and state
  InferenceConfig config_;
  bool initialized_;
  std::mutex inference_mutex_;

  // ORT backend executor
  std::unique_ptr<deep_ort_backend::OrtBackendExecutor> backend_executor_;

  // Helper methods
  cv::Mat preprocessImage(const cv::Mat & image);
  deep_ros::Tensor convertImagesToTensor(const std::vector<cv::Mat> & images);
  std::vector<std::vector<Detection>> postprocessOutput(
    const deep_ros::Tensor & output_tensor, const std::vector<cv::Size> & original_sizes);
  std::vector<Detection> processDetectionsForImage(
    const float * output_data,
    const std::vector<size_t> & output_shape,
    size_t image_index,
    const cv::Size & original_size);
  std::vector<Detection> applyNMS(const std::vector<Detection> & detections);
};

}  // namespace deep_object_detection
