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

#include "onnxruntime_gpu/onnxruntime_gpu_plugin.hpp"

#include <memory>
#include <string>
#include <vector>

#include <pluginlib/class_list_macros.hpp>

#include "deep_tensor/tensor.hpp"
#include "onnxruntime_gpu/onnx_cuda_allocator.hpp"

namespace onnxruntime_gpu
{

OnnxRuntimeGpuPlugin::OnnxRuntimeGpuPlugin(int device_id)
: device_id_(device_id)
, model_loaded_(false)
{
  // Try to create CUDA allocator, fallback to CPU if CUDA not available
  allocator_ = get_onnx_cuda_allocator(device_id);
  if (!allocator_) {
    // Fallback to CPU allocator if CUDA not available
    allocator_ = deep_ros::get_cpu_allocator();
  }

  // TODO(wato): Initialize ONNX Runtime session options with CUDA provider
}

bool OnnxRuntimeGpuPlugin::load_model(const std::filesystem::path & model_path)
{
  // TODO(wato): Implement actual ONNX model loading with CUDA provider
  // For now, just store the path and mark as loaded if file exists
  if (!std::filesystem::exists(model_path)) {
    return false;
  }

  current_model_path_ = model_path;
  model_loaded_ = true;

  // TODO(wato): Load ONNX model using ONNX Runtime API with CUDA provider
  // Ort::SessionOptions session_options;
  // session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{device_id_});
  // session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

  return true;
}

deep_ros::Tensor OnnxRuntimeGpuPlugin::inference(deep_ros::Tensor inputs)
{
  if (!model_loaded_) {
    throw std::runtime_error("No model loaded in ONNX Runtime GPU plugin");
  }

  // TODO(wato): Implement actual GPU inference
  // For now, return the input tensor as output
  return inputs;
}

void OnnxRuntimeGpuPlugin::unload_model()
{
  if (model_loaded_) {
    // TODO(wato): Cleanup ONNX Runtime session
    // session_.reset();
    model_loaded_ = false;
    current_model_path_.clear();
  }
}

std::string OnnxRuntimeGpuPlugin::backend_name() const
{
  return "onnxruntime-gpu";
}

std::vector<std::string> OnnxRuntimeGpuPlugin::supported_model_formats() const
{
  return {"onnx"};
}

std::shared_ptr<deep_ros::MemoryAllocator> OnnxRuntimeGpuPlugin::get_allocator() const
{
  return allocator_;
}

}  // namespace onnxruntime_gpu

// Export the plugin class
PLUGINLIB_EXPORT_CLASS(onnxruntime_gpu::OnnxRuntimeGpuPlugin, deep_ros::DeepBackendPlugin)
