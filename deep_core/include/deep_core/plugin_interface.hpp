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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "deep_core/types/tensor.hpp"

namespace deep_ros
{

enum class InferenceError
{
  MODEL_NOT_LOADED,
  INVALID_INPUT_SHAPE,
  INVALID_INPUT_TYPE,
  BACKEND_ERROR,
  OUT_OF_MEMORY
};

class InferenceResult
{
public:
  // Success constructor
  static InferenceResult success(std::vector<std::unique_ptr<Tensor>> tensors)
  {
    return InferenceResult(std::move(tensors));
  }

  // Error constructor
  static InferenceResult error(InferenceError error, const std::string & message)
  {
    return InferenceResult(error, message);
  }

  bool is_success() const
  {
    return !error_.has_value();
  }

  InferenceError error_code() const
  {
    if (!error_.has_value()) {
      throw std::runtime_error("Called error_code() on successful result");
    }
    return error_.value();
  }

  const std::string & error_message() const
  {
    if (!error_.has_value()) {
      throw std::runtime_error("Called error_message() on successful result");
    }
    return error_msg_;
  }

  std::vector<std::unique_ptr<Tensor>> & tensors()
  {
    if (error_.has_value()) {
      throw std::runtime_error("Called tensors() on failed result");
    }
    return tensors_;
  }

private:
  explicit InferenceResult(std::vector<std::unique_ptr<Tensor>> tensors)
  : tensors_(std::move(tensors))
  {}

  InferenceResult(InferenceError error, const std::string & message)
  : error_(error)
  , error_msg_(message)
  {}

  std::optional<InferenceError> error_;
  std::string error_msg_;
  std::vector<std::unique_ptr<Tensor>> tensors_;
};

// Pure C++ plugin interface - no ROS dependencies
class InferencePluginInterface
{
public:
  virtual ~InferencePluginInterface() = default;

  // Load model from file path
  virtual bool load_model(const std::filesystem::path & model_path) = 0;

  // Pure inference with detailed error handling
  virtual InferenceResult inference(std::vector<std::unique_ptr<Tensor>> inputs) = 0;

  // Cleanup
  virtual void unload_model() = 0;

  // Plugin metadata
  virtual std::string backend_name() const = 0;
  virtual std::vector<std::string> supported_model_formats() const = 0;
};

}  // namespace deep_ros
