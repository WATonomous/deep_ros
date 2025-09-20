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

#include "deep_backend/memory_allocator.hpp"
#include "deep_backend/inference_manager.hpp"

namespace deep_ros
{

// Pure C++ plugin interface - no ROS dependencies
class DeepBackendPlugin
{
public:
  virtual ~DeepBackendPlugin() = default;

  // Plugin metadata
  virtual std::string backend_name() const = 0;

  // Memory allocator provided by this plugin
  virtual std::shared_ptr<MemoryAllocator> get_allocator() const = 0;

  // Inference manager provided by this plugin
  virtual std::shared_ptr<InferenceManager> get_manager() const = 0;
};

}  // namespace deep_ros
