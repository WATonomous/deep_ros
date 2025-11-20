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

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "deep_ort_gpu_backend_plugin/ort_gpu_memory_allocator.hpp"

// Forward declaration in global scope
std::shared_ptr<deep_ros::BackendMemoryAllocator> get_simple_cpu_allocator();

namespace deep_ort_gpu_backend
{

OrtGpuBackendExecutor::OrtGpuBackendExecutor(int device_id, GpuExecutionProvider execution_provider)
: device_id_(device_id)
, execution_provider_(execution_provider)
, memory_info_(nullptr)
{
  // Initialize ORT environment
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_VERBOSE, "OrtGpuBackendExecutor");

  // Initialize session options
  session_options_ = std::make_unique<Ort::SessionOptions>();
  initialize_session_options();

  // Create memory info for GPU
  memory_info_ =
    Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtArenaAllocator, device_id_, OrtMemType::OrtMemTypeDefault);
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

GpuExecutionProvider OrtGpuBackendExecutor::get_execution_provider() const
{
  return execution_provider_;
}

bool OrtGpuBackendExecutor::load_model_impl(const std::filesystem::path & model_path)
{
  try {
    model_path_ = model_path;

    // Create session with GPU execution provider
    session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), *session_options_);

    return true;
  } catch (const std::exception & e) {
    std::cerr << "Failed to load model: " << e.what() << std::endl;
    session_.reset();
    return false;
  }
}

deep_ros::Tensor OrtGpuBackendExecutor::run_inference_impl(deep_ros::Tensor & input)
{
  if (!session_) {
    throw std::runtime_error("No model loaded");
  }

  try {
    std::cout << "Starting GPU inference..." << std::endl;

    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    auto output_name = session_->GetOutputNameAllocated(0, allocator);

    const char * input_names[] = {input_name.get()};
    const char * output_names[] = {output_name.get()};

    std::cout << "Input name: " << input_names[0] << ", Output name: " << output_names[0] << std::endl;

    // Validate input tensor shape
    std::cout << "Input tensor shape: [";
    for (size_t i = 0; i < input.shape().size(); ++i) {
      std::cout << input.shape()[i];
      if (i < input.shape().size() - 1) std::cout << ", ";
    }
    std::cout << "], dtype: " << static_cast<int>(input.dtype()) << std::endl;

    // Input tensor is already on CPU - create ONNXRuntime tensor directly
    std::cout << "Creating ONNXRuntime tensor from CPU input..." << std::endl;

    std::vector<int64_t> input_shape_int64(input.shape().begin(), input.shape().end());
    size_t total_elements = 1;
    for (size_t dim : input.shape()) {
      total_elements *= dim;
    }

    // Get ONNXRuntime's CPU allocator for proper memory management
    Ort::AllocatorWithDefaultOptions ort_allocator;

    // Let ONNXRuntime allocate and manage CPU memory completely
    Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(ort_allocator, input_shape_int64.data(), input_shape_int64.size());

    std::cout << "Created empty ONNXRuntime tensor, copying our data..." << std::endl;

    // Get ONNXRuntime's allocated memory and copy our data into it
    float * ort_input_data = input_tensor.GetTensorMutableData<float>();
    std::memcpy(ort_input_data, input.data(), total_elements * sizeof(float));

    std::cout << "Created ONNXRuntime input tensor, running inference..." << std::endl;

    // Run inference - ONNXRuntime will handle all GPU memory management internally
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    if (output_tensors.empty()) {
      throw std::runtime_error("Inference returned no outputs");
    }

    std::cout << "Inference completed, processing outputs..." << std::endl;

    // Get output tensor info
    auto & output_tensor = output_tensors[0];
    auto output_tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_tensor_info.GetShape();
    auto output_type = output_tensor_info.GetElementType();

    // Convert shape from int64_t to size_t
    std::vector<size_t> output_shape_sizet(output_shape.begin(), output_shape.end());

    std::cout << "Output shape: [";
    for (size_t i = 0; i < output_shape_sizet.size(); ++i) {
      std::cout << output_shape_sizet[i];
      if (i < output_shape_sizet.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Convert ONNX type back to deep_ros type
    deep_ros::DataType output_dtype;
    switch (output_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        output_dtype = deep_ros::DataType::FLOAT32;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        output_dtype = deep_ros::DataType::INT32;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        output_dtype = deep_ros::DataType::INT64;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        output_dtype = deep_ros::DataType::UINT8;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        output_dtype = deep_ros::DataType::INT8;
        break;
      default:
        throw std::runtime_error("Unsupported output data type from ONNX model");
    }

    // Get output data - ONNXRuntime manages this memory
    const float * output_data = output_tensor.GetTensorData<float>();
    size_t output_total_elements = 1;
    for (size_t dim : output_shape_sizet) {
      output_total_elements *= dim;
    }
    size_t output_bytes = output_total_elements * get_element_size(output_dtype);

    std::cout << "Creating output tensor with " << output_total_elements << " elements" << std::endl;

    // Create output tensor on CPU using our simple CPU allocator
    auto cpu_allocator = deep_ort_gpu_backend::get_simple_cpu_allocator();
    deep_ros::Tensor output_tensor_result(output_shape_sizet, output_dtype, cpu_allocator);

    std::cout << "Copying output data from ONNXRuntime to our CPU tensor..." << std::endl;

    // Simple memory copy - both are CPU memory
    std::memcpy(output_tensor_result.data(), output_data, output_bytes);

    std::cout << "GPU inference completed successfully" << std::endl;
    return output_tensor_result;
  } catch (const std::exception & e) {
    std::cerr << "GPU inference failed: " << e.what() << std::endl;
    throw std::runtime_error("Inference failed: " + std::string(e.what()));
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

  // Configure execution provider with fallback
  try {
    switch (execution_provider_) {
      case GpuExecutionProvider::CUDA:
        configure_cuda_provider();
        break;
      case GpuExecutionProvider::TENSORRT:
        try {
          configure_tensorrt_provider();
          std::cout << "TensorRT provider configured successfully" << std::endl;
        } catch (const std::exception & tensorrt_e) {
          std::cerr << "TensorRT failed during configuration, falling back to CUDA: " << tensorrt_e.what() << std::endl;
          execution_provider_ = GpuExecutionProvider::CUDA;
          configure_cuda_provider();
          std::cout << "Successfully fell back to CUDA provider" << std::endl;
        }
        break;
    }
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to configure any GPU execution provider: " + std::string(e.what()));
  }
}

void OrtGpuBackendExecutor::configure_cuda_provider()
{
  try {
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = device_id_;
    cuda_options.arena_extend_strategy = 0;  // kNextPowerOfTwo
    cuda_options.gpu_mem_limit = 0;  // Use default
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.do_copy_in_default_stream = 1;

    session_options_->AppendExecutionProvider_CUDA(cuda_options);
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to configure CUDA provider: " + std::string(e.what()));
  }
}

void OrtGpuBackendExecutor::configure_tensorrt_provider()
{
  try {
    if (!check_tensorrt_dependencies()) {
      throw std::runtime_error("TensorRT dependencies not satisfied");
    }

    std::cout << "All TensorRT dependencies found, configuring TensorRT provider..." << std::endl;

    int cuda_runtime_version = 0;
    // Removed: cudaRuntimeGetVersion(&cuda_runtime_version);
    // std::cout << "CUDA runtime version detected: " << cuda_runtime_version << std::endl;

    // Use string-based provider configuration for better compatibility
    std::unordered_map<std::string, std::string> tensorrt_options;
    tensorrt_options["device_id"] = std::to_string(device_id_);
    tensorrt_options["trt_max_workspace_size"] = "67108864";  // 64MB
    tensorrt_options["trt_max_partition_iterations"] = "1";
    tensorrt_options["trt_min_subgraph_size"] = "1";
    tensorrt_options["trt_fp16_enable"] = "0";
    tensorrt_options["trt_engine_cache_enable"] = "0";

    std::cout << "Attempting TensorRT provider registration..." << std::endl;
    session_options_->AppendExecutionProvider("NvTensorRTRTXExecutionProvider", tensorrt_options);
    std::cout << "TensorRT provider registered successfully" << std::endl;
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

std::vector<size_t> OrtGpuBackendExecutor::get_output_shape(const std::vector<size_t> & input_shape) const
{
  if (!session_) {
    throw std::runtime_error("No model loaded");
  }

  try {
    // Get output tensor info from the session
    auto output_type_info = session_->GetOutputTypeInfo(0);
    auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    std::vector<size_t> output_shape;
    for (int64_t dim : shape) {
      if (dim == -1) {
        // Dynamic dimension - use corresponding input dimension
        output_shape.push_back(input_shape[output_shape.size()]);
      } else {
        output_shape.push_back(static_cast<size_t>(dim));
      }
    }

    return output_shape;
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to get output shape: " + std::string(e.what()));
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

bool OrtGpuBackendExecutor::verify_gpu_availability() const
{
  return true;
}

void OrtGpuBackendExecutor::set_device() const
{}

bool OrtGpuBackendExecutor::check_tensorrt_dependencies() const
{
  // Since libraries are properly registered with ldconfig, we can rely on system resolution
  std::cout << "Checking TensorRT dependencies..." << std::endl;

  // Check for core TensorRT library (try exact version first)
  void * handle = dlopen("libnvinfer.so.10", RTLD_LAZY);
  if (!handle) {
    // Try lean version as fallback
    handle = dlopen("libnvinfer_lean.so.10", RTLD_LAZY);
    if (handle) {
      dlclose(handle);
      std::cout << "Found TensorRT lean library (v10)" << std::endl;
      // For lean version, plugin library is not required
      return true;
    }

    std::cerr << "TensorRT core library not found. Error: " << dlerror() << std::endl;
    return false;
  } else {
    dlclose(handle);
    std::cout << "Found TensorRT full library (v10)" << std::endl;
  }

  // For full TensorRT, check plugin library
  handle = dlopen("libnvinfer_plugin.so.10", RTLD_LAZY);
  if (!handle) {
    std::cerr << "Warning: TensorRT plugin library not found. Error: " << dlerror() << std::endl;
    // Don't fail completely - some models don't need plugins
  } else {
    dlclose(handle);
    std::cout << "Found TensorRT plugin library (v10)" << std::endl;
  }

  // Check for ONNX parser library (required by ONNXRuntime TensorRT provider)
  handle = dlopen("libnvonnxparser.so.10", RTLD_LAZY);
  if (!handle) {
    std::cerr << "TensorRT ONNX parser library not found. Error: " << dlerror() << std::endl;
    std::cerr << "Install with: sudo apt install libnvonnxparser10" << std::endl;
    return false;
  } else {
    dlclose(handle);
    std::cout << "Found TensorRT ONNX parser library (v10)" << std::endl;
  }

  std::cout << "All TensorRT dependencies satisfied" << std::endl;
  return true;
}

}  // namespace deep_ort_gpu_backend
