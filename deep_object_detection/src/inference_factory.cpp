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

#include <memory>

#include "deep_object_detection/inference_interface.hpp"
// #include "deep_object_detection/onnx_inference.hpp"
#include "deep_object_detection/ort_backend_inference.hpp"

namespace deep_object_detection
{

std::unique_ptr<InferenceInterface> createInferenceEngine(const InferenceConfig & config)
{
  switch (config.backend) {
    case InferenceBackend::ORT_BACKEND:
      return std::make_unique<OrtBackendInference>(config);

    case InferenceBackend::AUTO:
    default:
      // Fall back to ONNX inference if else fails
      return std::make_unique<OrtBackendInference>(config);
  }
}

}  // namespace deep_object_detection
