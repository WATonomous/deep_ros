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

#include "onnxruntime/onnxruntime_plugin.hpp"

#include <memory>
#include <string>
#include <vector>

#include <pluginlib/class_list_macros.hpp>

#include "deep_tensor/tensor.hpp"
#include "onnxruntime/onnx_cpu_allocator.hpp"

namespace onnxruntime
{

OnnxRuntimePlugin::OnnxRuntimePlugin()
: allocator_(get_onnx_cpu_allocator())
, model_loaded_(false)
{
  // TODO(wato): Initialize ONNX Runtime session options and environment
}

bool OnnxRuntimePlugin::load_model(const std::filesystem::path & model_path)
{
  // TODO(wato): Implement actual ONNX model loading
  // For now, just store the path and mark as loaded if file exists
  if (!std::filesystem::exists(model_path)) {
    return false;
  }

  current_model_path_ = model_path;
  model_loaded_ = true;

  // TODO(wato): Load ONNX model using ONNX Runtime API
  // session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

  return true;
}

deep_ros::Tensor OnnxRuntimePlugin::inference(deep_ros::Tensor inputs)
{
  if (!model_loaded_) {
    throw std::runtime_error("No model loaded in ONNX Runtime plugin");
  }

  // TODO(wato): Implement actual inference
  // For now, return the input tensor as output
  return inputs;
}

void OnnxRuntimePlugin::unload_model()
{
  if (model_loaded_) {
    // TODO(wato): Cleanup ONNX Runtime session
    // session_.reset();
    model_loaded_ = false;
    current_model_path_.clear();
  }
}

std::string OnnxRuntimePlugin::backend_name() const
{
  return "onnxruntime";
}

std::vector<std::string> OnnxRuntimePlugin::supported_model_formats() const
{
  return {"onnx"};
}

std::shared_ptr<deep_ros::MemoryAllocator> OnnxRuntimePlugin::get_allocator() const
{
  return allocator_;
}

}  // namespace onnxruntime

// Export the plugin class
PLUGINLIB_EXPORT_CLASS(onnxruntime::OnnxRuntimePlugin, deep_ros::DeepBackendPlugin)
