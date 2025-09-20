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

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <deep_core/plugin_interfaces/backend_inference_executor.hpp>
#include <deep_core/types/tensor.hpp>
#include <onnxruntime_cxx_api.h>

namespace deep_ort_backend
{

/**
 * @brief ONNX Runtime backend inference executor
 * 
 * Provides inference execution using ONNX Runtime with CPU optimization.
 */
class OrtBackendExecutor : public deep_ros::BackendInferenceExecutor
{
public:
  OrtBackendExecutor();
  ~OrtBackendExecutor() override = default;

  bool load_model(const std::filesystem::path & model_path) override;
  deep_ros::Tensor run_inference(deep_ros::Tensor input) override;
  void unload_model() override;
  std::vector<std::string> supported_model_formats() const override;

private:
  bool model_loaded_{false};
  std::filesystem::path model_path_;
  
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;
  
  ONNXTensorElementDataType convert_to_onnx_type(deep_ros::DataType dtype) const;
  std::vector<size_t> get_output_shape(const std::vector<size_t>& input_shape) const;
  size_t get_element_size(deep_ros::DataType dtype) const;
};

}  // namespace deep_ort_backend