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

#include <deep_core/plugin_interfaces/backend_inference_executor.hpp>
#include <deep_core/types/tensor.hpp>
#include <opencv2/opencv.hpp>

#include "deep_object_detection/inference_interface.hpp"

namespace deep_object_detection
{

/**
 * @brief Backend type enumeration
 */
enum class BackendType
{
  CPU,  ///< Use CPU backend (deep_ort_backend_plugin)
  GPU  ///< Use GPU backend (deep_ort_gpu_backend_plugin)
};

/**
 * @brief ORT Backend inference adapter for object detection
 *
 * This class adapts both CPU and GPU ORT backend executors to work with
 * the object detection inference interface, providing zero-copy inference
 * capabilities while maintaining compatibility with OpenCV images.
 *
 * The backend type can be configured to use either CPU or GPU execution.
 */
class OrtBackendInference : public InferenceInterface
{
public:
  /**
   * @brief Constructor with backend type selection
   * @param config Inference configuration
   * @param backend_type Backend type (CPU or GPU)
   * @param device_id GPU device ID (only used for GPU backend, default: 0)
   */
  explicit OrtBackendInference(
    const InferenceConfig & config, BackendType backend_type = BackendType::GPU, int device_id = 0);
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

  /**
   * @brief Get current backend type
   * @return Current backend type (CPU or GPU)
   */
  BackendType getBackendType() const
  {
    return backend_type_;
  }

  /**
   * @brief Get GPU device ID (only relevant for GPU backend)
   * @return GPU device ID
   */
  int getDeviceId() const
  {
    return device_id_;
  }

private:
  // Configuration and state
  InferenceConfig config_;
  BackendType backend_type_;
  int device_id_;
  bool initialized_;
  std::mutex inference_mutex_;

  // Backend components (using base interfaces for polymorphism)
  std::shared_ptr<deep_ros::BackendInferenceExecutor> backend_executor_;
  std::shared_ptr<deep_ros::BackendMemoryAllocator> backend_allocator_;

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
