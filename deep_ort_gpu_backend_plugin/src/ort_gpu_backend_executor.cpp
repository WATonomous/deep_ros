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

#include "deep_ort_gpu_backend_plugin/ort_gpu_backend_executor.hpp"

#include <dlfcn.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "deep_ort_gpu_backend_plugin/ort_gpu_memory_allocator.hpp"

// Forward declarations
std::shared_ptr<deep_ros::BackendMemoryAllocator> get_simple_cpu_allocator();

namespace deep_ort_gpu_backend
{
// Forward declaration for cast
class OrtGpuCpuMemoryAllocator;

OrtGpuBackendExecutor::OrtGpuBackendExecutor(
  int device_id,
  const std::string & execution_provider,
  const rclcpp::Logger & logger,
  bool enable_trt_engine_cache,
  std::string trt_engine_cache_path)
: device_id_(device_id)
, execution_provider_(execution_provider)
, logger_(logger)
, enable_trt_engine_cache_(enable_trt_engine_cache)
, trt_engine_cache_path_(std::move(trt_engine_cache_path))
, memory_info_(nullptr)
{
  // Initialize ORT environment with minimal logging
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "OrtGpuBackendExecutor");

  // Initialize session options
  session_options_ = std::make_unique<Ort::SessionOptions>();
  initialize_session_options();

  // Create memory info for GPU
  memory_info_ =
    Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtArenaAllocator, device_id_, OrtMemType::OrtMemTypeDefault);

  // Register our custom CPU allocator for output tensors
  auto custom_allocator_shared = get_ort_gpu_cpu_allocator();
  custom_allocator_ = custom_allocator_shared;
}

OrtGpuBackendExecutor::~OrtGpuBackendExecutor()
{}

std::vector<std::string> OrtGpuBackendExecutor::supported_model_formats() const
{
  return {"onnx"};
}

int OrtGpuBackendExecutor::get_device_id() const
{
  return device_id_;
}

bool OrtGpuBackendExecutor::load_model_impl(const std::filesystem::path & model_path)
{
  try {
    model_path_ = model_path;

    // Create session with GPU execution provider
    session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), *session_options_);

    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(logger_, "Failed to load model: %s", e.what());
    session_.reset();
    return false;
  }
}

deep_ros::Tensor OrtGpuBackendExecutor::run_inference_impl(deep_ros::Tensor & input)
{
  if (!session_) {
    throw std::runtime_error("No ONNX session available");
  }
  // for logging
  try {
    const bool using_tensorrt = is_tensorrt_provider();
    if (using_tensorrt && !tensorrt_engine_ready_ && !tensorrt_engine_build_in_progress_) {
      tensorrt_engine_build_in_progress_ = true;
      tensorrt_engine_build_start_ = std::chrono::steady_clock::now();
      RCLCPP_INFO(
        logger_,
        "TensorRT engine build started (model=%s, batch_size=%zu)",
        model_path_.filename().c_str(),
        input.shape().empty() ? 0U : input.shape().front());
    }

    // Convert deep_ros::DataType to ONNX tensor element type
    ONNXTensorElementDataType onnx_type = convert_to_onnx_type(input.dtype());
    std::vector<int64_t> input_shape_int64(input.shape().begin(), input.shape().end());

    // Get our custom allocator for output binding
    auto custom_allocator_shared = get_ort_gpu_cpu_allocator();
    auto * custom_allocator = static_cast<OrtGpuCpuMemoryAllocator *>(custom_allocator_shared.get());

    // Create input tensor wrapping existing input memory
    size_t input_size_bytes = input.byte_size();
    auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value ort_input = Ort::Value::CreateTensor(
      cpu_memory_info, input.data(), input_size_bytes, input_shape_int64.data(), input_shape_int64.size(), onnx_type);

    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    auto output_name = session_->GetOutputNameAllocated(0, allocator);

    // Create IO binding for inference
    Ort::IoBinding binding(*session_);
    binding.BindInput(input_name.get(), ort_input);

    // Bind output to use our custom allocator
    binding.BindOutput(output_name.get(), custom_allocator->get_ort_memory_info());

    // Run inference with IO binding
    Ort::RunOptions run_options;
    session_->Run(run_options, binding);

    // Get output values allocated by ONNX Runtime
    Ort::AllocatorWithDefaultOptions default_allocator;
    std::vector<Ort::Value> output_tensors = binding.GetOutputValues(default_allocator);

    if (output_tensors.empty()) {
      throw std::runtime_error("TensorRT inference returned no outputs");
    }

    auto & output_tensor = output_tensors.front();
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape_int64 = output_info.GetShape();

    std::vector<size_t> output_shape;
    output_shape.reserve(output_shape_int64.size());
    for (int64_t dim : output_shape_int64) {
      if (dim <= 0) {
        throw std::runtime_error("Invalid runtime output dimension: " + std::to_string(dim));
      }
      output_shape.push_back(static_cast<size_t>(dim));
    }

    void * output_data = output_tensor.GetTensorMutableData<void>();

    // Create deep_ros tensor wrapping the ONNX-allocated memory
    deep_ros::Tensor output(output_data, output_shape, input.dtype());

    if (tensorrt_engine_build_in_progress_) {
      const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - tensorrt_engine_build_start_);
      RCLCPP_INFO(logger_, "TensorRT engine build completed in %.2f seconds", elapsed.count());
      tensorrt_engine_build_in_progress_ = false;
      tensorrt_engine_ready_ = true;
    }

    return output;
  } catch (const std::exception & e) {
    tensorrt_engine_build_in_progress_ = false;
    throw std::runtime_error("GPU inference failed: " + std::string(e.what()));
  }
}

void OrtGpuBackendExecutor::unload_model_impl()
{
  session_.reset();
  model_path_.clear();
}

void OrtGpuBackendExecutor::initialize_session_options()
{
  // Set basic options
  session_options_->SetIntraOpNumThreads(1);
  session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Configure execution provider based on string
  normalized_provider_ = execution_provider_;
  std::transform(normalized_provider_.begin(), normalized_provider_.end(), normalized_provider_.begin(), ::tolower);

  if (normalized_provider_ == "cuda") {
    configure_cuda_provider();
  } else if (normalized_provider_ == "tensorrt") {
    configure_tensorrt_provider();
  } else {
    throw std::runtime_error(
      "Unknown execution provider: " + execution_provider_ + ". Valid options: 'cuda', 'tensorrt'");
  }
}

void OrtGpuBackendExecutor::configure_cuda_provider()
{
  try {
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = device_id_;
    cuda_options.arena_extend_strategy = 0;  // kNextPowerOfTwo
    cuda_options.gpu_mem_limit = 0;  // No memory limit
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cuda_options.do_copy_in_default_stream = 1;

    session_options_->AppendExecutionProvider_CUDA(cuda_options);

    // Set optimization level
    session_options_->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options_->EnableMemPattern();
    session_options_->EnableCpuMemArena();
    session_options_->DisableProfiling();

    // Use environment allocators
    session_options_->AddConfigEntry("session.use_env_allocators", "1");
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to configure CUDA provider: " + std::string(e.what()));
  }
}

void OrtGpuBackendExecutor::configure_tensorrt_provider()
{
  try {
    setenv("CUDA_MODULE_LOADING", "LAZY", 1);

    OrtTensorRTProviderOptions tensorrt_options{};
    tensorrt_options.device_id = device_id_;
    tensorrt_options.trt_max_workspace_size = 67108864;  // 64MB
    tensorrt_options.trt_max_partition_iterations = 1;
    tensorrt_options.trt_min_subgraph_size = 1;
    tensorrt_options.trt_engine_cache_enable = enable_trt_engine_cache_ ? 1 : 0;
    tensorrt_options.trt_force_sequential_engine_build = 1;
    tensorrt_options.trt_fp16_enable = 0;
    tensorrt_options.trt_int8_enable = 0;
    tensorrt_options.has_user_compute_stream = 0;
    tensorrt_options.user_compute_stream = nullptr;

    std::string cache_path = trt_engine_cache_path_;
    if (enable_trt_engine_cache_) {
      if (cache_path.empty()) {
        cache_path = "/tmp/deep_ros_ort_trt_cache";
      }
      std::error_code ec;
      std::filesystem::create_directories(cache_path, ec);
      if (ec) {
        RCLCPP_WARN(logger_, "Failed to create TensorRT cache directory %s: %s", cache_path.c_str(), ec.message().c_str());
      }
      trt_engine_cache_path_ = cache_path;
      tensorrt_options.trt_engine_cache_path = trt_engine_cache_path_.c_str();
    }

    RCLCPP_INFO(logger_, "Configuring TensorRT execution provider on device %d", device_id_);
    session_options_->AppendExecutionProvider_TensorRT(tensorrt_options);
    RCLCPP_INFO(logger_, "TensorRT provider registered successfully");
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to configure TensorRT provider: " + std::string(e.what()));
  }
}

ONNXTensorElementDataType OrtGpuBackendExecutor::convert_to_onnx_type(deep_ros::DataType dtype) const
{
  switch (dtype) {
    case deep_ros::DataType::FLOAT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case deep_ros::DataType::INT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case deep_ros::DataType::INT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case deep_ros::DataType::UINT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case deep_ros::DataType::INT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

size_t OrtGpuBackendExecutor::get_element_size(deep_ros::DataType dtype) const
{
  switch (dtype) {
    case deep_ros::DataType::FLOAT32:
      return sizeof(float);
    case deep_ros::DataType::INT32:
      return sizeof(int32_t);
    case deep_ros::DataType::INT64:
      return sizeof(int64_t);
    case deep_ros::DataType::UINT8:
      return sizeof(uint8_t);
    case deep_ros::DataType::INT8:
      return sizeof(int8_t);
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

bool OrtGpuBackendExecutor::is_tensorrt_provider() const
{
  return normalized_provider_ == "tensorrt";
}

}  // namespace deep_ort_gpu_backend
