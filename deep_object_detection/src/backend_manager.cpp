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
#include <atomic>
#include <cctype>
#include <stdexcept>
#include <utility>

#include <deep_core/plugin_interfaces/deep_backend_plugin.hpp>
#include <pluginlib/class_loader.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>

namespace deep_object_detection
{

BackendManager::BackendManager(rclcpp_lifecycle::LifecycleNode & node, const DetectionParams & params)
: node_(node)
, params_(params)
, plugin_loader_(std::make_unique<pluginlib::ClassLoader<deep_ros::DeepBackendPlugin>>("deep_core", "deep_ros::DeepBackendPlugin"))
{}

BackendManager::~BackendManager()
{
  // Reset plugin holder before plugin loader is destroyed
  // This ensures proper cleanup order to avoid class_loader warnings
  plugin_holder_.reset();
  executor_.reset();
  allocator_.reset();
}

void BackendManager::buildProviderOrder()
{
  auto normalize_pref = params_.preferred_provider;
  std::transform(normalize_pref.begin(), normalize_pref.end(), normalize_pref.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  provider_order_.clear();
  if (normalize_pref == "cpu") {
    provider_order_.push_back(Provider::CPU);
  } else if (normalize_pref == "cuda") {
    provider_order_.push_back(Provider::CUDA);
    provider_order_.push_back(Provider::CPU);
  } else {
    provider_order_.push_back(Provider::TENSORRT);
    provider_order_.push_back(Provider::CUDA);
    provider_order_.push_back(Provider::CPU);
  }
  active_provider_index_ = 0;
}

bool BackendManager::initialize(size_t start_index)
{
  return initializeBackend(start_index);
}

deep_ros::Tensor BackendManager::infer(const PackedInput & input)
{
  if (!executor_) {
    throw std::runtime_error("No backend initialized; dropping batch");
  }

  size_t attempts = 0;
  const size_t max_attempts = provider_order_.empty() ? 1 : provider_order_.size();
  while (attempts < max_attempts) {
    auto input_tensor = buildInputTensor(input);
    try {
      return executor_->run_inference(input_tensor);
    } catch (const std::exception & e) {
      RCLCPP_WARN(node_.get_logger(), "Inference failed on provider %s: %s", active_provider_.c_str(), e.what());
    }

    ++attempts;
    if (!fallbackToNextProvider()) {
      break;
    }
  }

  throw std::runtime_error("All backends failed");
}

std::vector<deep_ros::Tensor> BackendManager::inferAllOutputs(const PackedInput & input)
{
  if (!executor_) {
    throw std::runtime_error("No backend initialized; dropping batch");
  }

  // TODO: When backend executor interface supports multiple outputs, enhance this
  // For now, return a vector with the single output for backward compatibility
  // The postprocessor can still handle multi-output configuration when available
  size_t attempts = 0;
  const size_t max_attempts = provider_order_.empty() ? 1 : provider_order_.size();
  while (attempts < max_attempts) {
    auto input_tensor = buildInputTensor(input);
    try {
      auto output = executor_->run_inference(input_tensor);
      return {output};  // Return as single-element vector
    } catch (const std::exception & e) {
      RCLCPP_WARN(node_.get_logger(), "Inference failed on provider %s: %s", active_provider_.c_str(), e.what());
    }

    ++attempts;
    if (!fallbackToNextProvider()) {
      break;
    }
  }

  throw std::runtime_error("All backends failed");
}

bool BackendManager::fallbackToNextProvider()
{
  if (provider_order_.empty()) {
    return false;
  }

  const size_t next_index = active_provider_index_ + 1;
  if (next_index >= provider_order_.size()) {
    RCLCPP_ERROR(node_.get_logger(), "No more providers to fall back to");
    return false;
  }

  active_provider_index_ = next_index;
  RCLCPP_WARN(
    node_.get_logger(),
    "Falling back to provider: %s",
    providerToString(provider_order_[active_provider_index_]).c_str());
  return initializeBackend(active_provider_index_);
}

bool BackendManager::initializeBackend(size_t start_index)
{
  for (size_t idx = start_index; idx < provider_order_.size(); ++idx) {
    active_provider_index_ = idx;
    Provider provider = provider_order_[idx];

    if ((provider == Provider::TENSORRT || provider == Provider::CUDA) && !isCudaRuntimeAvailable()) {
      RCLCPP_WARN(
        node_.get_logger(),
        "Skipping provider %s: CUDA runtime libraries not found (libcudart/libcuda).",
        providerToString(provider).c_str());
      continue;
    }

    try {
      const std::string plugin_name = providerToPluginName(provider);
      if (plugin_name.empty()) {
        RCLCPP_WARN(node_.get_logger(), "No plugin name for provider: %s", providerToString(provider).c_str());
        continue;
      }

      // Create backend config node with provider-specific parameters
      rclcpp_lifecycle::LifecycleNode::SharedPtr backend_node;
      if (provider == Provider::TENSORRT || provider == Provider::CUDA) {
        const auto provider_name = providerToString(provider);
        auto overrides = std::vector<rclcpp::Parameter>{
          rclcpp::Parameter("Backend.device_id", params_.device_id),
          rclcpp::Parameter("Backend.execution_provider", provider_name),
          rclcpp::Parameter("Backend.trt_engine_cache_enable", params_.enable_trt_engine_cache),
          rclcpp::Parameter("Backend.trt_engine_cache_path", params_.trt_engine_cache_path)};
        backend_node = createBackendConfigNode(provider_name, std::move(overrides));
      } else {
        backend_node = createBackendConfigNode("cpu");
      }

      // Load plugin using pluginlib
      plugin_holder_ = plugin_loader_->createUniqueInstance(plugin_name);
      plugin_holder_->initialize(backend_node);
      allocator_ = plugin_holder_->get_allocator();
      executor_ = plugin_holder_->get_inference_executor();

      if (!executor_ || !allocator_) {
        throw std::runtime_error("Executor or allocator is null");
      }

      if (params_.model_path.empty()) {
        throw std::runtime_error("Model path is not set");
      }

      if (!executor_->load_model(params_.model_path)) {
        throw std::runtime_error("Failed to load model: " + params_.model_path);
      }

      active_provider_ = providerToString(provider);
      declareActiveProviderParameter(active_provider_);
      warmupTensorShapeCache(provider);

      RCLCPP_INFO(
        node_.get_logger(),
        "Initialized backend using provider: %s (device %d)",
        active_provider_.c_str(),
        params_.device_id);
      return true;
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        node_.get_logger(), "Provider %s initialization failed: %s", providerToString(provider).c_str(), e.what());
    }
  }

  RCLCPP_ERROR(node_.get_logger(), "Unable to initialize any execution provider");
  return false;
}

void BackendManager::warmupTensorShapeCache(Provider provider)
{
  if (!params_.warmup_tensor_shapes) {
    return;
  }
  if (provider == Provider::CPU) {
    return;
  }
  if (!executor_ || !allocator_) {
    return;
  }
  const size_t channels = 3;
  const size_t height = static_cast<size_t>(params_.preprocessing.input_height);
  const size_t width = static_cast<size_t>(params_.preprocessing.input_width);
  const size_t per_image = channels * height * width;

  const int batch = params_.batch_size_limit;
  RCLCPP_INFO(
    node_.get_logger(),
    "Priming %s backend tensor shapes for batch size %d",
    providerToString(provider).c_str(),
    batch);

  PackedInput dummy;
  dummy.shape = {static_cast<size_t>(batch), channels, height, width};
  dummy.data.assign(static_cast<size_t>(batch) * per_image, 0.0f);

  auto input_tensor = buildInputTensor(dummy);
  try {
    (void)executor_->run_inference(input_tensor);
    RCLCPP_DEBUG(node_.get_logger(), "Cached tensor shape for batch size %d", batch);
  } catch (const std::exception & e) {
    RCLCPP_WARN(node_.get_logger(), "Warmup inference for batch size %d failed: %s", batch, e.what());
  }
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

rclcpp_lifecycle::LifecycleNode::SharedPtr BackendManager::createBackendConfigNode(
  const std::string & suffix, std::vector<rclcpp::Parameter> overrides) const
{
  static std::atomic<uint64_t> backend_node_counter{0};
  rclcpp::NodeOptions options;
  if (!overrides.empty()) {
    options.parameter_overrides(overrides);
  }
  options.start_parameter_services(false);
  options.start_parameter_event_publisher(false);

  const auto node_id = backend_node_counter.fetch_add(1, std::memory_order_relaxed);
  auto node_name = "detection_backend_" + suffix + "_" + std::to_string(node_id);
  return std::make_shared<rclcpp_lifecycle::LifecycleNode>(node_name, options);
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
    throw std::runtime_error("No backend executor or allocator available");
  }

  try {
    // Create a dummy input tensor to run inference and get output shape
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
  } catch (const std::exception & e) {
    RCLCPP_WARN(
      node_.get_logger(),
      "Failed to get output shape from model via dummy inference: %s. Will use auto-detection.",
      e.what());
    return {};  // Return empty to trigger auto-detection
  }
}

}  // namespace deep_object_detection

