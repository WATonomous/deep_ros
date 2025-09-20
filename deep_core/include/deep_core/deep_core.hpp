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

/**
 * @file deep_core.hpp
 * @brief Convenience header that includes all deep_core components
 * 
 * This header provides a single include for all deep_core functionality:
 * - Core types (Tensor, DataType)
 * - Plugin interfaces (BackendMemoryAllocator, BackendInferenceExecutor, DeepBackendPlugin)
 * - Base node classes (DeepNodeBase)
 * 
 * Usage:
 * @code
 * #include <deep_core/deep_core.hpp>
 * @endcode
 */

// Core types
#include "deep_core/types/tensor.hpp"

// Plugin interfaces
#include "deep_core/plugin_interfaces/backend_memory_allocator.hpp"
#include "deep_core/plugin_interfaces/backend_inference_executor.hpp"
#include "deep_core/plugin_interfaces/deep_backend_plugin.hpp"

// Base classes
#include "deep_core/deep_node_base.hpp"

// Optional: convenience namespace alias
namespace deep = deep_ros;
