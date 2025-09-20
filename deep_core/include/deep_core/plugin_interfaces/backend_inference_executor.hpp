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
   */
  virtual bool load_model(const std::filesystem::path & model_path) = 0;

  /**
   * @brief Run inference on input tensor
   * @param input Input tensor
   * @return Output tensor
   */
  virtual Tensor run_inference(Tensor input) = 0;

  /**
   * @brief Unload the currently loaded model
   */
  virtual void unload_model() = 0;

  /**
   * @brief Get supported model formats
   * @return Vector of supported formats (e.g., "onnx", "pb")
   */
  virtual std::vector<std::string> supported_model_formats() const = 0;
};

}  // namespace deep_ros
