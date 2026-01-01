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
#include <stdexcept>
#include <string>
#include <vector>

#include "deep_object_detection/detection_types.hpp"
#include "deep_object_detection/generic_postprocessor.hpp"

namespace deep_object_detection
{

/**
 * @brief Factory class for creating detection postprocessors
 *
 * Creates the appropriate postprocessor based on configuration parameters.
 */
class PostprocessorFactory
{
public:
  /**
   * @brief Create a postprocessor based on configuration
   *
   * @param params Detection parameters containing postprocessing config
   * @param output_shape Optional output shape from model for auto-detection
   * @return Unique pointer to the created postprocessor
   * @throws std::runtime_error if postprocessor type is unsupported
   */
  static std::unique_ptr<GenericPostprocessor> create(
    const DetectionParams & params,
    const std::vector<size_t> & output_shape = {})
  {
    const auto & config = params.postprocessing;
    const auto & model_meta = params.model_metadata;
    const bool use_letterbox = (params.preprocessing.resize_method == ResizeMethod::LETTERBOX);

    // Always use generic postprocessor for model-agnostic operation
    GenericPostprocessor::OutputLayout layout;
    if (!output_shape.empty()) {
      layout = GenericPostprocessor::detectLayout(output_shape);
    } else {
      layout.auto_detect = true;
    }
    return std::make_unique<GenericPostprocessor>(
      config, layout, model_meta.bbox_format, model_meta.num_classes,
      params.class_names, use_letterbox);
  }

  /**
   * @brief Create a postprocessor based on string type
   *
   * @param type_str String identifier for postprocessor type ("auto" or "generic" for auto-detection)
   * @param params Detection parameters
   * @param output_shape Optional output shape for auto-detection
   * @return Unique pointer to the created postprocessor
   */
  static std::unique_ptr<GenericPostprocessor> createFromString(
    const std::string & /*type_str*/,
    DetectionParams & params,
    const std::vector<size_t> & output_shape = {})
  {
    // Always use generic postprocessor - type_str parameter kept for API compatibility only
    return create(params, output_shape);
  }
};

}  // namespace deep_object_detection

