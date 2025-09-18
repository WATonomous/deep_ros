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

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <mutex>

#include "deep_object_detection/inference_interface.hpp"

namespace deep_object_detection
{

/**
 * @brief ONNX Runtime implementation of the inference interface
 */
class ONNXInference : public InferenceInterface
{
public:
  explicit ONNXInference(const InferenceConfig & config);
  ~ONNXInference() override;

  bool initialize() override;
  std::vector<std::vector<Detection>> inferBatch(const std::vector<cv::Mat> & images) override;
  std::vector<Detection> infer(const cv::Mat & image) override;

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
  InferenceConfig config_;
  bool initialized_;
  std::mutex inference_mutex_;

  // ONNX Runtime objects
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;

  // Model info
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;

  // Helper methods
  bool loadModel();
  void setupProviders();
  cv::Mat preprocessImage(const cv::Mat & image);
  std::vector<Detection> postprocessOutputs(const std::vector<Ort::Value> & outputs, const cv::Size & original_size);
  std::vector<Detection> applyNMS(const std::vector<Detection> & detections);
  void logModelInfo();
};

}  // namespace deep_object_detection
