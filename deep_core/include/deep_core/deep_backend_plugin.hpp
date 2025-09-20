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

#include "deep_tensor/memory_allocator.hpp"
#include "deep_tensor/tensor.hpp"

namespace deep_ros
{

// Pure C++ plugin interface - no ROS dependencies
class DeepBackendPlugin
{
public:
  virtual ~DeepBackendPlugin() = default;

  // Load model from file path
  virtual bool load_model(const std::filesystem::path & model_path) = 0;

  // Pure inference with detailed error handling
  virtual Tensor inference(Tensor inputs) = 0;

  // Cleanup
  virtual void unload_model() = 0;

  // Plugin metadata
  virtual std::string backend_name() const = 0;
  virtual std::vector<std::string> supported_model_formats() const = 0;

  // Memory allocator provided by this plugin
  virtual std::shared_ptr<MemoryAllocator> get_allocator() const = 0;
};

}  // namespace deep_ros
