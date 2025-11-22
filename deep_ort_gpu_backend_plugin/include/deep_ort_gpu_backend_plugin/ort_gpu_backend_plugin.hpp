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
#include <string>

#include <deep_core/plugin_interfaces/backend_inference_executor.hpp>
#include <deep_core/plugin_interfaces/backend_memory_allocator.hpp>
#include <deep_core/plugin_interfaces/deep_backend_plugin.hpp>

#include "deep_ort_gpu_backend_plugin/ort_gpu_backend_executor.hpp"

namespace deep_ort_gpu_backend
{

// Forward declarations
enum class GpuExecutionProvider;

/**
 * @brief ONNX Runtime GPU backend plugin
 *
 * Combines ORT GPU memory allocator and inference executor into a single
 * backend plugin for use with pluginlib. Supports CUDA
 * execution provider and possible more in the future.
 */
class OrtGpuBackendPlugin : public deep_ros::DeepBackendPlugin
{
public:
  /**
   * @brief Constructor - initializes ORT GPU allocator and executor
   * @param device_id CUDA device ID (default: 0)
   * @param execution_provider GPU execution provider (default: CUDA)
   */
  explicit OrtGpuBackendPlugin(int device_id = 0, GpuExecutionProvider execution_provider = GpuExecutionProvider::CUDA);

  /**
   * @brief Destructor
   */
  ~OrtGpuBackendPlugin() override = default;

  /**
   * @brief Get backend name
   * @return "onnxruntime_gpu"
   */
  std::string backend_name() const override;

  /**
   * @brief Get the ORT GPU memory allocator
   * @return Shared pointer to ORT GPU memory allocator
   */
  std::shared_ptr<deep_ros::BackendMemoryAllocator> get_allocator() const override;

  /**
   * @brief Get the ORT GPU inference executor
   * @return Shared pointer to ORT GPU inference executor
   */
  std::shared_ptr<deep_ros::BackendInferenceExecutor> get_inference_executor() const override;

  /**
   * @brief Get CUDA device ID
   * @return CUDA device ID
   */
  int get_device_id() const;

  /**
   * @brief Get current execution provider
   * @return Current GPU execution provider
   */
  GpuExecutionProvider get_execution_provider() const;

private:
  int device_id_;
  GpuExecutionProvider execution_provider_;
  std::shared_ptr<deep_ros::BackendMemoryAllocator> allocator_;
  std::shared_ptr<deep_ros::BackendInferenceExecutor> executor_;

  /**
   * @brief Initialize GPU components
   * @return true if successful, false otherwise
   */
  bool initialize_gpu_components();
};

}  // namespace deep_ort_gpu_backend
