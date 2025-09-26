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

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <deep_core/plugin_interfaces/backend_inference_executor.hpp>
#include <deep_core/types/tensor.hpp>

namespace deep_ort_backend
{

/**
 * @brief ONNX Runtime backend inference executor
 *
 * Provides inference execution using ONNX Runtime with CPU optimization.
 * Uses zero-copy IO binding for efficient tensor operations.
 */
class OrtBackendExecutor : public deep_ros::BackendInferenceExecutor
{
public:
  /**
   * @brief Constructor - initializes ONNX Runtime environment
   */
  OrtBackendExecutor();

  /**
   * @brief Destructor
   */
  ~OrtBackendExecutor() override = default;

  /**
   * @brief Get supported model formats
   * @return Vector containing "onnx"
   */
  std::vector<std::string> supported_model_formats() const override;

protected:
  /**
   * @brief Load an ONNX model from file
   * @param model_path Path to the .onnx model file
   * @return true if successful, false otherwise
   */
  bool load_model_impl(const std::filesystem::path & model_path) override;

  /**
   * @brief Run inference using zero-copy IO binding
   * @param input Input tensor (must be compatible with model input)
   * @return Output tensor with inference results
   * @throws std::runtime_error if inference fails or no model loaded
   */
  deep_ros::Tensor run_inference_impl(deep_ros::Tensor & input) override;

  /**
   * @brief Unload the currently loaded model
   */
  void unload_model_impl() override;

private:
  std::filesystem::path model_path_;

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;

  /**
   * @brief Convert deep_ros DataType to ONNX tensor element type
   * @param dtype deep_ros data type
   * @return ONNX tensor element data type
   */
  ONNXTensorElementDataType convert_to_onnx_type(deep_ros::DataType dtype) const;

  /**
   * @brief Get model output shape based on input shape
   * @param input_shape Input tensor shape
   * @return Expected output tensor shape
   * @throws std::runtime_error if model not loaded or shape inference fails
   */
  std::vector<size_t> get_output_shape(const std::vector<size_t> & input_shape) const;

  /**
   * @brief Get element size in bytes for a data type
   * @param dtype Data type
   * @return Size in bytes per element
   */
  size_t get_element_size(deep_ros::DataType dtype) const;
};

}  // namespace deep_ort_backend
