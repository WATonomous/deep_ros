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

#include "deep_core/plugin_interfaces/backend_inference_executor.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace deep_ros
{

bool BackendInferenceExecutor::load_model(const std::filesystem::path & model_path)
{
  // Validate path
  if (model_path.empty()) {
    throw std::invalid_argument("Empty model path provided to load_model");
  }

  // Call implementation and track state
  bool success = load_model_impl(model_path);
  model_loaded_ = success;
  return success;
}

Tensor BackendInferenceExecutor::run_inference(const Tensor & input)
{
  // Validate input tensor
  if (input.data() == nullptr) {
    throw std::invalid_argument("Null data pointer in input tensor");
  }

  if (input.size() == 0) {
    throw std::invalid_argument("Empty tensor provided to run_inference");
  }

  // Check for valid shape
  if (input.shape().empty()) {
    throw std::invalid_argument("Tensor has empty shape");
  }

  // Check if model is loaded
  if (!model_loaded_) {
    throw std::runtime_error("No model loaded for inference");
  }

  return run_inference_impl(input);
}

void BackendInferenceExecutor::unload_model()
{
  unload_model_impl();
  model_loaded_ = false;
}

}  // namespace deep_ros
