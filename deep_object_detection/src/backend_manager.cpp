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

#include "deep_object_detection/backend_manager.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <deep_core/plugin_interfaces/deep_backend_plugin.hpp>
#include <pluginlib/class_loader.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>

namespace deep_object_detection
{

BackendManager::BackendManager(rclcpp_lifecycle::LifecycleNode & node, const DetectionParams & params)
: node_(node)
, params_(params)
, plugin_loader_(
    std::make_unique<pluginlib::ClassLoader<deep_ros::DeepBackendPlugin>>("deep_core", "deep_ros::DeepBackendPlugin"))
{}

BackendManager::~BackendManager()
{
  plugin_holder_.reset();
  executor_.reset();
  allocator_.reset();
}

void BackendManager::initialize()
{
  provider_ = parseProvider(params_.preferred_provider);
  initializeBackend();
}

deep_ros::Tensor BackendManager::infer(const PackedInput & input)
{
  if (!executor_) {
    RCLCPP_ERROR(node_.get_logger(), "Cannot perform inference: no backend executor initialized");
    throw std::runtime_error("Cannot perform inference: no backend executor initialized");
  }

  auto input_tensor = buildInputTensor(input);
  return executor_->run_inference(input_tensor);
}

std::vector<deep_ros::Tensor> BackendManager::inferAllOutputs(const PackedInput & input)
{
  if (!executor_) {
    RCLCPP_ERROR(node_.get_logger(), "Cannot perform inference: no backend executor initialized");
    throw std::runtime_error("Cannot perform inference: no backend executor initialized");
  }

  auto input_tensor = buildInputTensor(input);
  auto output = executor_->run_inference(input_tensor);
  return {output};
}

Provider BackendManager::parseProvider(const std::string & provider_str) const
{
  std::string normalized = provider_str;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (normalized == "cpu") {
    return Provider::CPU;
  } else if (normalized == "cuda") {
    return Provider::CUDA;
  } else if (normalized == "tensorrt") {
    return Provider::TENSORRT;
  } else {
    throw std::runtime_error("Unknown provider: " + provider_str + ". Valid options: cpu, cuda, tensorrt");
  }
}

void BackendManager::initializeBackend()
{
  if ((provider_ == Provider::TENSORRT || provider_ == Provider::CUDA) && !isCudaRuntimeAvailable()) {
    std::string error = "Provider " + providerToString(provider_) +
                        " requires CUDA runtime libraries (libcudart/libcuda) which are not available";
    throw std::runtime_error(error);
  }

  const std::string plugin_name = providerToPluginName(provider_);
  if (plugin_name.empty()) {
    throw std::runtime_error("No plugin name for provider: " + providerToString(provider_));
  }

  const auto provider_name = providerToString(provider_);

  if (!node_.has_parameter("Backend.execution_provider")) {
    node_.declare_parameter<std::string>("Backend.execution_provider", provider_name);
  } else {
    node_.set_parameters({rclcpp::Parameter("Backend.execution_provider", provider_name)});
  }

  if (!node_.has_parameter("Backend.device_id")) {
    node_.declare_parameter<int>("Backend.device_id", params_.device_id);
  } else {
    node_.set_parameters({rclcpp::Parameter("Backend.device_id", params_.device_id)});
  }

  if (!node_.has_parameter("Backend.trt_engine_cache_enable")) {
    node_.declare_parameter<bool>("Backend.trt_engine_cache_enable", params_.enable_trt_engine_cache);
  } else {
    node_.set_parameters({rclcpp::Parameter("Backend.trt_engine_cache_enable", params_.enable_trt_engine_cache)});
  }

  if (!node_.has_parameter("Backend.trt_engine_cache_path")) {
    node_.declare_parameter<std::string>("Backend.trt_engine_cache_path", params_.trt_engine_cache_path);
  } else {
    node_.set_parameters({rclcpp::Parameter("Backend.trt_engine_cache_path", params_.trt_engine_cache_path)});
  }

  auto node_ptr = node_.shared_from_this();
  plugin_holder_ = plugin_loader_->createUniqueInstance(plugin_name);
  plugin_holder_->initialize(node_ptr);
  allocator_ = plugin_holder_->get_allocator();
  executor_ = plugin_holder_->get_inference_executor();

  if (!executor_ || !allocator_) {
    throw std::runtime_error("Executor or allocator is null after plugin initialization");
  }

  if (params_.model_path.empty()) {
    throw std::runtime_error("Model path is not set");
  }

  if (!executor_->load_model(params_.model_path)) {
    throw std::runtime_error("Failed to load model: " + params_.model_path);
  }

  active_provider_ = providerToString(provider_);
  declareActiveProviderParameter(active_provider_);
  warmupTensorShapeCache();

  RCLCPP_INFO(
    node_.get_logger(),
    "Initialized backend using provider: %s (device %d)",
    active_provider_.c_str(),
    params_.device_id);
}

void BackendManager::warmupTensorShapeCache()
{
  if (!params_.warmup_tensor_shapes) {
    return;
  }
  if (provider_ == Provider::CPU) {
    return;
  }
  if (!executor_ || !allocator_) {
    return;
  }
  const size_t channels = RGB_CHANNELS;
  const size_t height = static_cast<size_t>(params_.preprocessing.input_height);
  const size_t width = static_cast<size_t>(params_.preprocessing.input_width);
  const size_t per_image = channels * height * width;

  // Use batch size 1 for warmup - actual batch size will be determined by MultiImage
  const size_t batch = 1;
  RCLCPP_INFO(
    node_.get_logger(), "Priming %s backend tensor shapes for batch size %zu", active_provider_.c_str(), batch);

  PackedInput dummy;
  dummy.shape = {batch, channels, height, width};
  dummy.data.assign(batch * per_image, 0.0f);

  auto input_tensor = buildInputTensor(dummy);
  (void)executor_->run_inference(input_tensor);
  RCLCPP_DEBUG(node_.get_logger(), "Cached tensor shape for batch size %zu", batch);
}

bool BackendManager::isCudaRuntimeAvailable() const
{
  const char * libs[] = {"libcudart.so.12", "libcudart.so.11", "libcudart.so", "libcuda.so.1", "libcuda.so"};
  for (const auto * lib : libs) {
    void * handle = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
    if (handle) {
      dlclose(handle);
      return true;
    }
  }
  return false;
}

std::string BackendManager::providerToString(Provider provider) const
{
  switch (provider) {
    case Provider::TENSORRT:
      return "tensorrt";
    case Provider::CUDA:
      return "cuda";
    case Provider::CPU:
      return "cpu";
    default:
      return "unknown";
  }
}

void BackendManager::declareActiveProviderParameter(const std::string & value)
{
  rcl_interfaces::msg::ParameterDescriptor desc;
  desc.read_only = true;
  desc.description = "Current execution provider in use (read-only)";

  if (node_.has_parameter("active_provider")) {
    (void)node_.set_parameters({rclcpp::Parameter("active_provider", value)});
  } else {
    node_.declare_parameter<std::string>("active_provider", value, desc);
  }
}

deep_ros::Tensor BackendManager::buildInputTensor(const PackedInput & packed) const
{
  deep_ros::Tensor tensor(packed.shape, deep_ros::DataType::FLOAT32, allocator_);
  const size_t bytes = packed.data.size() * sizeof(float);
  allocator_->copy_from_host(tensor.data(), packed.data.data(), bytes);
  return tensor;
}

std::string BackendManager::providerToPluginName(Provider provider) const
{
  switch (provider) {
    case Provider::TENSORRT:
    case Provider::CUDA:
      return "onnxruntime_gpu";
    case Provider::CPU:
      return "onnxruntime_cpu";
    default:
      return "";
  }
}

std::vector<size_t> BackendManager::getOutputShape(const std::vector<size_t> & input_shape) const
{
  if (!executor_ || !allocator_) {
    RCLCPP_ERROR(
      node_.get_logger(),
      "Cannot get output shape: backend not initialized (executor: %s, allocator: %s)",
      executor_ ? "available" : "null",
      allocator_ ? "available" : "null");
    throw std::runtime_error("Cannot get output shape: backend executor or allocator not available");
  }

  // NO TRY-CATCH - let exceptions propagate
  PackedInput dummy;
  dummy.shape = input_shape;
  size_t total_elements = 1;
  for (size_t dim : input_shape) {
    total_elements *= dim;
  }
  dummy.data.assign(total_elements, 0.0f);

  auto input_tensor = buildInputTensor(dummy);
  auto output_tensor = executor_->run_inference(input_tensor);
  return output_tensor.shape();
}

}  // namespace deep_object_detection
