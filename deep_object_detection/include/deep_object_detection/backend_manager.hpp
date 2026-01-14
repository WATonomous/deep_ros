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

/**
 * @file backend_manager.hpp
 * @brief Backend plugin manager for ONNX Runtime inference
 *
 * This header defines the BackendManager class which:
 * - Loads ONNX Runtime backend plugins (CPU, CUDA, TensorRT)
 * - Initializes models with selected execution providers
 * - Provides memory allocators for tensor operations
 * - Executes inference on batched input tensors
 * - Supports fail-fast behavior (no silent fallbacks)
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <deep_core/plugin_interfaces/deep_backend_plugin.hpp>
#include <deep_core/types/tensor.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "deep_object_detection/detection_types.hpp"

namespace deep_ros
{
class BackendInferenceExecutor;
class BackendMemoryAllocator;
}  // namespace deep_ros

namespace deep_object_detection
{

/**
 * @brief Manages backend plugin loading and inference execution
 *
 * Handles loading of ONNX Runtime backend plugins (CPU, CUDA, TensorRT),
 * model loading, and inference execution. Supports fail-fast behavior:
 * if the requested provider is unavailable, initialization fails immediately.
 *
 * The manager:
 * - Loads the appropriate backend plugin based on preferred_provider
 * - Initializes the model with the selected execution provider
 * - Provides memory allocator for tensor operations
 * - Executes inference on batched input tensors
 * - Warms up tensor shape cache for optimal performance
 */
class BackendManager
{
public:
  /**
   * @brief Construct the backend manager
   * @param node Lifecycle node reference for parameter access and logging
   * @param params Detection parameters containing model path and provider settings
   *
   * Does not initialize the backend. Call initialize() after construction.
   */
  BackendManager(rclcpp_lifecycle::LifecycleNode & node, const DetectionParams & params);

  /**
   * @brief Destructor: releases plugin resources
   */
  ~BackendManager();

  /**
   * @brief Initialize the backend: load plugin, model, and warmup
   *
   * Loads the backend plugin, initializes the model with the selected execution
   * provider, and warms up tensor shape cache. Throws if provider is unavailable
   * or model loading fails (fail-fast behavior).
   */
  void initialize();

  /**
   * @brief Run inference on a batched input tensor
   * @param input Packed input tensor (batched preprocessed images)
   * @return Output tensor from model inference
   *
   * Converts PackedInput to deep_ros::Tensor and runs inference.
   * Returns the first output tensor (for single-output models).
   */
  deep_ros::Tensor infer(const PackedInput & input);

  /**
   * @brief Run inference and return all output tensors
   * @param input Packed input tensor (batched preprocessed images)
   * @return Vector of all output tensors from model
   *
   * Useful for multi-output models (e.g., separate boxes, scores, classes).
   */
  std::vector<deep_ros::Tensor> inferAllOutputs(const PackedInput & input);

  /**
   * @brief Get the active execution provider name
   * @return String name of the active provider ("tensorrt", "cuda", or "cpu")
   */
  const std::string & activeProvider() const
  {
    return active_provider_;
  }

  /**
   * @brief Get the memory allocator for tensor operations
   * @return Shared pointer to backend memory allocator
   *
   * Used by preprocessor to allocate tensors in backend-appropriate memory
   * (e.g., GPU memory for CUDA/TensorRT, CPU memory for CPU provider).
   */
  std::shared_ptr<deep_ros::BackendMemoryAllocator> allocator() const
  {
    return allocator_;
  }

  /**
   * @brief Check if executor is initialized
   * @return True if executor is available, false otherwise
   */
  bool hasExecutor() const
  {
    return static_cast<bool>(executor_);
  }

  /**
   * @brief Get output tensor shape for given input shape
   * @param input_shape Input tensor shape [batch, channels, height, width]
   * @return Output tensor shape
   *
   * Uses cached shape information from warmup. Returns shape of first output
   * tensor for the given input shape.
   */
  std::vector<size_t> getOutputShape(const std::vector<size_t> & input_shape) const;

private:
  /**
   * @brief Initialize the backend plugin and load model
   *
   * Loads the plugin using pluginlib, creates executor and allocator,
   * and loads the ONNX model. Throws if provider is unavailable or model
   * loading fails.
   */
  void initializeBackend();

  /**
   * @brief Warmup tensor shape cache for optimal performance
   *
   * Runs dummy inference with expected input shapes to warm up TensorRT
   * engine cache and optimize memory allocation. Always enabled (not configurable).
   */
  void warmupTensorShapeCache();

  /**
   * @brief Parse provider string to Provider enum
   * @param provider_str Provider name ("tensorrt", "cuda", or "cpu")
   * @return Provider enum value
   */
  Provider parseProvider(const std::string & provider_str) const;

  /**
   * @brief Check if CUDA runtime is available
   * @return True if CUDA is available, false otherwise
   *
   * Used to validate CUDA/TensorRT provider requirements.
   */
  bool isCudaRuntimeAvailable() const;

  /**
   * @brief Convert Provider enum to string
   * @param provider Provider enum value
   * @return String representation ("tensorrt", "cuda", or "cpu")
   */
  std::string providerToString(Provider provider) const;

  /**
   * @brief Declare active_provider parameter for runtime visibility
   * @param value Provider name to declare
   *
   * Makes the active provider visible in ROS parameter server for debugging.
   */
  void declareActiveProviderParameter(const std::string & value);

  /**
   * @brief Build input tensor from PackedInput
   * @param packed Packed input data (flattened float array and shape)
   * @return Tensor ready for inference
   *
   * Converts PackedInput (CPU memory) to deep_ros::Tensor using the
   * backend allocator (may copy to GPU memory for CUDA/TensorRT).
   */
  deep_ros::Tensor buildInputTensor(const PackedInput & packed) const;

  /// Reference to lifecycle node (for parameters and logging)
  rclcpp_lifecycle::LifecycleNode & node_;
  /// Detection parameters (model path, provider settings)
  const DetectionParams & params_;
  /// Selected execution provider enum
  Provider provider_;
  /// Active provider name (for logging and parameter server)
  std::string active_provider_{"unknown"};
  /// Inference executor (runs model inference)
  std::shared_ptr<deep_ros::BackendInferenceExecutor> executor_;
  /// Memory allocator (for tensor allocation)
  std::shared_ptr<deep_ros::BackendMemoryAllocator> allocator_;
  /// Plugin loader (loads backend plugin via pluginlib)
  std::unique_ptr<pluginlib::ClassLoader<deep_ros::DeepBackendPlugin>> plugin_loader_;
  /// Plugin instance (holds loaded plugin)
  pluginlib::UniquePtr<deep_ros::DeepBackendPlugin> plugin_holder_;

  /**
   * @brief Map Provider enum to plugin name
   * @param provider Provider enum value
   * @return Plugin name for pluginlib (e.g., "deep_ort_backend_plugin::OrtBackendPlugin")
   */
  std::string providerToPluginName(Provider provider) const;
};

}  // namespace deep_object_detection
