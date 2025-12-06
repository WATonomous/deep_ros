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

#include "deep_yolo_inference/backend_manager.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <stdexcept>
#include <utility>

#include <deep_core/plugin_interfaces/deep_backend_plugin.hpp>
#include <deep_ort_backend_plugin/ort_backend_plugin.hpp>
#include <deep_ort_gpu_backend_plugin/ort_gpu_backend_plugin.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>

namespace deep_yolo_inference
{

BackendManager::BackendManager(rclcpp::Node & node, const YoloParams & params)
: node_(node)
, params_(params)
{}

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
      switch (provider) {
        case Provider::TENSORRT:
        case Provider::CUDA: {
          const auto provider_name = providerToString(provider);
          auto overrides = std::vector<rclcpp::Parameter>{
            rclcpp::Parameter("Backend.device_id", params_.device_id),
            rclcpp::Parameter("Backend.execution_provider", provider_name),
            rclcpp::Parameter("Backend.trt_engine_cache_enable", params_.enable_trt_engine_cache),
            rclcpp::Parameter("Backend.trt_engine_cache_path", params_.trt_engine_cache_path)};
          auto backend_node = createBackendConfigNode(provider_name, std::move(overrides));
          auto plugin = std::make_shared<deep_ort_gpu_backend::OrtGpuBackendPlugin>();
          plugin->initialize(backend_node);
          allocator_ = plugin->get_allocator();
          executor_ = plugin->get_inference_executor();
          plugin_holder_ = plugin;
          break;
        }
        case Provider::CPU: {
          auto backend_node = createBackendConfigNode("cpu");
          auto plugin = std::make_shared<deep_ort_backend::OrtBackendPlugin>();
          plugin->initialize(backend_node);
          allocator_ = plugin->get_allocator();
          executor_ = plugin->get_inference_executor();
          plugin_holder_ = plugin;
          break;
        }
      }

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
  const size_t height = static_cast<size_t>(params_.input_height);
  const size_t width = static_cast<size_t>(params_.input_width);
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
  auto node_name = "yolo_backend_" + suffix + "_" + std::to_string(node_id);
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

}  // namespace deep_yolo_inference
