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

#include <rclcpp/rclcpp.hpp>

#include "deep_core/types/tensor.hpp"

namespace deep_ros
{

/**
 * @brief Backend plugin interface for inference execution
 *
 * This plugin interface allows different backends to provide
 * their inference execution implementations.
 */
class BackendInferenceExecutor
{
public:
  virtual ~BackendInferenceExecutor() = default;

  /**
   * @brief Load a model from file
   * @param model_path Path to the model file
   * @return true if successful, false otherwise
   * @throws std::invalid_argument if model_path is empty or file doesn't exist
   */
  bool load_model(const std::filesystem::path & model_path);

  /**
   * @brief Run inference on input tensor
   * @param input Input tensor (note: some backends may require mutable access for zero-copy operations)
   * @return Output tensor
   * @throws std::invalid_argument if input tensor is invalid
   * @throws std::runtime_error if no model is loaded
   */
  Tensor run_inference(Tensor & input);

  /**
   * @brief Unload the currently loaded model
   */
  void unload_model();

  /**
   * @brief Check if a model is currently loaded
   * @return true if model is loaded, false otherwise
   */
  bool is_model_loaded() const
  {
    return model_loaded_;
  }

  /**
   * @brief Get supported model formats
   * @return Vector of supported formats (e.g., "onnx", "pb")
   */
  virtual std::vector<std::string> supported_model_formats() const = 0;

protected:
  /**
   * @brief Implementation of load_model (to be overridden by backends)
   */
  virtual bool load_model_impl(const std::filesystem::path & model_path) = 0;

  /**
   * @brief Implementation of run_inference (to be overridden by backends)
   * @param input Input tensor (note: some backends may require mutable access)
   */
  virtual Tensor run_inference_impl(Tensor & input) = 0;

  /**
   * @brief Implementation of unload_model (to be overridden by backends)
   */
  virtual void unload_model_impl() = 0;

private:
  bool model_loaded_ = false;
};

}  // namespace deep_ros
