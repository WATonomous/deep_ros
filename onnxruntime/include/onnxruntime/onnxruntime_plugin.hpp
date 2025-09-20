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

#include "deep_core/deep_backend_plugin.hpp"
#include "deep_tensor/memory_allocator.hpp"

namespace onnxruntime
{

/**
 * @brief ONNX Runtime CPU inference plugin
 *
 * Provides CPU-based inference using ONNX Runtime with optimized memory allocation.
 */
class OnnxRuntimePlugin : public deep_ros::DeepBackendPlugin
{
public:
  OnnxRuntimePlugin();
  ~OnnxRuntimePlugin() override = default;

  // DeepBackendPlugin implementation
  bool load_model(const std::filesystem::path & model_path) override;
  deep_ros::Tensor inference(deep_ros::Tensor inputs) override;
  void unload_model() override;
  std::string backend_name() const override;
  std::vector<std::string> supported_model_formats() const override;
  std::shared_ptr<deep_ros::MemoryAllocator> get_allocator() const override;

private:
  std::shared_ptr<deep_ros::MemoryAllocator> allocator_;
  bool model_loaded_;
  std::filesystem::path current_model_path_;

  // TODO(wato): Add ONNX Runtime session and related members when implementing inference
};

}  // namespace onnxruntime
