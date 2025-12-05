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

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <deep_core/plugin_interfaces/backend_inference_executor.hpp>
#include <deep_core/types/tensor.hpp>

namespace deep_ort_gpu_backend
{

/**
 * @brief Execution provider type for GPU inference
 */
enum class GpuExecutionProvider
{
  CUDA,  ///< NVIDIA CUDA execution provider
  TENSORRT  ///< NVIDIA TensorRT execution provider
};

/**
 * @brief ONNX Runtime GPU backend inference executor
 *
 * Provides inference execution using ONNX Runtime with GPU optimization
 * supporting both CUDA and TensorRT execution providers.
 * Uses zero-copy IO binding for efficient tensor operations.
 */
class OrtGpuBackendExecutor : public deep_ros::BackendInferenceExecutor
{
public:
  /**
   * @brief Constructor - initializes ONNX Runtime environment with GPU support
   * @param device_id CUDA device ID (default: 0)
   * @param execution_provider GPU execution provider (default: CUDA)
   */
  explicit OrtGpuBackendExecutor(
    int device_id = 0, GpuExecutionProvider execution_provider = GpuExecutionProvider::CUDA);

  /**
   * @brief Destructor
   */
  ~OrtGpuBackendExecutor();

  /**
   * @brief Get supported model formats
   * @return Vector containing "onnx"
   */
  std::vector<std::string> supported_model_formats() const override;

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

protected:
  /**
   * @brief Load an ONNX model from file with GPU optimization
   * @param model_path Path to the .onnx model file
   * @return true if successful, false otherwise
   */
  bool load_model_impl(const std::filesystem::path & model_path) override;

  /**
   * @brief Run inference with GPU acceleration
   * @param input Input tensor (must be compatible with model input)
   * @return Output tensor with inference results
   * @throws std::runtime_error if inference fails or no model loaded
   */
  deep_ros::Tensor run_inference_impl(deep_ros::Tensor & input) override;

  /**
   * @brief Unload the currently loaded model
   */
  void unload_model_impl() override;

private:
  std::filesystem::path model_path_;
  int device_id_;
  GpuExecutionProvider execution_provider_;
  void * persistent_cuda_ptr_ = nullptr;
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  Ort::MemoryInfo memory_info_;
  std::shared_ptr<deep_ros::BackendMemoryAllocator> custom_allocator_;

  /**
   * @brief Initialize session options with GPU execution provider
   */
  void initialize_session_options();

  /**
   * @brief Configure CUDA execution provider
   */
  void configure_cuda_provider();

  /**
   * @brief Configure TensorRT execution provider
   */
  void configure_tensorrt_provider();

  /**
   * @brief Convert deep_ros DataType to ONNX tensor element type
   * @param dtype deep_ros data type
   * @return ONNX tensor element data type
   */
  ONNXTensorElementDataType convert_to_onnx_type(deep_ros::DataType dtype) const;

  /**
   * @brief Get model output shape based on input shape
   * @param input_shape Input tensor shape
   * @return Expected output tensor shape
   * @throws std::runtime_error if model not loaded or shape inference fails
   */
  std::vector<size_t> get_output_shape(const std::vector<size_t> & input_shape) const;

  /**
   * @brief Get element size in bytes for a data type
   * @param dtype Data type
   * @return Size in bytes per element
   */
  size_t get_element_size(deep_ros::DataType dtype) const;

  /**
   * @brief Verify GPU availability and compatibility
   * @return true if GPU is available and compatible
   */
  bool verify_gpu_availability() const;

  /**
   * @brief Set CUDA device context
   */
  void set_device() const;
};

}  // namespace deep_ort_gpu_backend
