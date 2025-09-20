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

#include <stdexcept>

#include "deep_ort_backend_plugin/ort_cpu_memory_allocator.hpp"

namespace deep_ort_backend
{

OrtBackendExecutor::OrtBackendExecutor()
: memory_info_(nullptr)
{
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "deep_ort_backend");
  memory_info_ = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
}

bool OrtBackendExecutor::load_model(const std::filesystem::path & model_path)
{
  if (!std::filesystem::exists(model_path)) {
    return false;
  }

  try {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    model_path_ = model_path;
    model_loaded_ = true;
    return true;
  } catch (const std::exception & e) {
    model_loaded_ = false;
    return false;
  }
}

deep_ros::Tensor OrtBackendExecutor::run_inference(deep_ros::Tensor input)
{
  if (!model_loaded_) {
    throw std::runtime_error("No model loaded for inference");
  }

  if (!session_) {
    throw std::runtime_error("No ONNX session available");
  }

  try {
    // Convert deep_ros::DataType to ONNX tensor element type
    ONNXTensorElementDataType onnx_type = convert_to_onnx_type(input.dtype());

    // Create input OrtValue that wraps the input tensor's memory (zero-copy!)
    size_t input_size_bytes = input.size() * get_element_size(input.dtype());
    std::vector<int64_t> input_shape_int64(input.shape().begin(), input.shape().end());

    Ort::Value ort_input = Ort::Value::CreateTensor(
      memory_info_, input.data(), input_size_bytes, input_shape_int64.data(), input_shape_int64.size(), onnx_type);

    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    auto output_name = session_->GetOutputNameAllocated(0, allocator);

    // Get output shape (assuming we know it or can infer it)
    auto output_shape = get_output_shape(input.shape());

    // Allocate output tensor using our custom allocator
    auto tensor_allocator = get_ort_cpu_allocator();
    deep_ros::Tensor output(output_shape, input.dtype(), tensor_allocator);

    // Create output OrtValue that wraps the output tensor's memory (zero-copy!)
    size_t output_size_bytes = output.size() * get_element_size(output.dtype());
    std::vector<int64_t> output_shape_int64(output.shape().begin(), output.shape().end());

    Ort::Value ort_output = Ort::Value::CreateTensor(
      memory_info_, output.data(), output_size_bytes, output_shape_int64.data(), output_shape_int64.size(), onnx_type);

    // Create IO binding for zero-copy inference
    Ort::IoBinding binding(*session_);
    binding.BindInput(input_name.get(), ort_input);
    binding.BindOutput(output_name.get(), ort_output);

    // Run inference with IO binding (zero-copy!)
    Ort::RunOptions run_options;
    session_->Run(run_options, binding);

    return output;

  } catch (const std::exception & e) {
    throw std::runtime_error("ONNX Runtime inference failed: " + std::string(e.what()));
  }
}

void OrtBackendExecutor::unload_model()
{
  if (model_loaded_) {
    session_.reset();
    model_loaded_ = false;
    model_path_.clear();
  }
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
    case deep_ros::DataType::INT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case deep_ros::DataType::INT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case deep_ros::DataType::UINT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
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
  switch (dtype) {
    case deep_ros::DataType::FLOAT32:
      return sizeof(float);
    case deep_ros::DataType::INT32:
      return sizeof(int32_t);
    case deep_ros::DataType::INT64:
      return sizeof(int64_t);
    case deep_ros::DataType::UINT8:
      return sizeof(uint8_t);
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

}  // namespace deep_ort_backend
