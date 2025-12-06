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

#include "deep_ort_backend_plugin/ort_backend_executor.hpp"

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "deep_ort_backend_plugin/ort_cpu_memory_allocator.hpp"

namespace deep_ort_backend
{

OrtBackendExecutor::OrtBackendExecutor()
: memory_info_(nullptr)
{
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "deep_ort_backend");
  memory_info_ = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  // Register our custom allocator with the environment
  auto custom_allocator_shared = get_ort_cpu_allocator();
  auto * custom_allocator = static_cast<OrtCpuMemoryAllocator *>(custom_allocator_shared.get());
  OrtStatus * status =
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->RegisterAllocator(*env_, custom_allocator->get_ort_allocator());
  if (status != nullptr) {
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
    // Log warning but don't fail - we can still work with default allocator
  }
}

bool OrtBackendExecutor::load_model_impl(const std::filesystem::path & model_path)
{
  if (!std::filesystem::exists(model_path)) {
    throw std::runtime_error("Model file not found: " + model_path.string());
  }

  try {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Configure session to use environment allocators (our custom allocator)
    session_options.AddConfigEntry("session.use_env_allocators", "1");

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    model_path_ = model_path;
    return true;
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
  }
}

deep_ros::Tensor OrtBackendExecutor::run_inference_impl(deep_ros::Tensor & input)
{
  if (!session_) {
    throw std::runtime_error("No ONNX session available");
  }

  try {
    // Convert deep_ros::DataType to ONNX tensor element type
    ONNXTensorElementDataType onnx_type = convert_to_onnx_type(input.dtype());
    std::vector<int64_t> input_shape_int64(input.shape().begin(), input.shape().end());

    auto custom_allocator_shared = get_ort_cpu_allocator();

    // Create input tensor that wraps existing input memory (zero-copy!)
    size_t input_size_bytes = input.byte_size();
    Ort::Value ort_input = Ort::Value::CreateTensor(
      memory_info_, input.data(), input_size_bytes, input_shape_int64.data(), input_shape_int64.size(), onnx_type);

    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    auto output_name = session_->GetOutputNameAllocated(0, allocator);
    const char * input_names[] = {input_name.get()};
    const char * output_names[] = {output_name.get()};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names, &ort_input, 1, output_names, 1);

    if (output_tensors.empty()) {
      throw std::runtime_error("ONNX Runtime inference returned no outputs");
    }

    auto & output_tensor = output_tensors.front();
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape_int64 = output_info.GetShape();
    std::vector<size_t> output_shape(output_shape_int64.begin(), output_shape_int64.end());

    deep_ros::Tensor output(output_shape, input.dtype(), custom_allocator_shared);
    std::memcpy(output.data(), output_tensor.GetTensorData<float>(), output.byte_size());

    return output;
  } catch (const std::exception & e) {
    throw std::runtime_error("ONNX Runtime inference failed: " + std::string(e.what()));
  }
}

void OrtBackendExecutor::unload_model_impl()
{
  session_.reset();
  model_path_.clear();
}

std::vector<std::string> OrtBackendExecutor::supported_model_formats() const
{
  return {"onnx"};
}

ONNXTensorElementDataType OrtBackendExecutor::convert_to_onnx_type(deep_ros::DataType dtype) const
{
  switch (dtype) {
    case deep_ros::DataType::FLOAT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case deep_ros::DataType::FLOAT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case deep_ros::DataType::INT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case deep_ros::DataType::INT16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case deep_ros::DataType::INT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case deep_ros::DataType::INT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case deep_ros::DataType::UINT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case deep_ros::DataType::UINT16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case deep_ros::DataType::UINT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case deep_ros::DataType::UINT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case deep_ros::DataType::BOOL:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:
      throw std::runtime_error("Unsupported data type for ONNX Runtime");
  }
}

std::vector<size_t> OrtBackendExecutor::get_output_shape(const std::vector<size_t> & input_shape) const
{
  if (!session_) {
    throw std::runtime_error("No session available to query output shape");
  }

  try {
    // Get output type info for the first output (assuming single output model)
    auto output_type_info = session_->GetOutputTypeInfo(0);
    auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

    // Get the output shape from the model
    auto output_shape_int64 = tensor_info.GetShape();

    // Convert int64_t shape to size_t and handle dynamic dimensions
    std::vector<size_t> output_shape;
    output_shape.reserve(output_shape_int64.size());

    for (size_t i = 0; i < output_shape_int64.size(); ++i) {
      int64_t dim = output_shape_int64[i];

      if (dim == -1) {
        // Dynamic dimension: use corresponding input dimension
        if (i < input_shape.size()) {
          output_shape.push_back(input_shape[i]);
        } else {
          throw std::runtime_error("Cannot resolve dynamic output dimension " + std::to_string(i));
        }
      } else if (dim <= 0) {
        throw std::runtime_error("Invalid output dimension: " + std::to_string(dim));
      } else {
        output_shape.push_back(static_cast<size_t>(dim));
      }
    }

    return output_shape;
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to get output shape from model: " + std::string(e.what()));
  }
}

size_t OrtBackendExecutor::get_element_size(deep_ros::DataType dtype) const
{
  return deep_ros::get_dtype_size(dtype);
}

}  // namespace deep_ort_backend
