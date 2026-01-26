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

#include "deep_object_detection/deep_object_detection_node.hpp"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <deep_core/plugin_interfaces/deep_backend_plugin.hpp>
#include <deep_core/types/tensor.hpp>
#include <deep_msgs/msg/multi_image.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>

#include "deep_object_detection/detection_types.hpp"
#include "deep_object_detection/generic_postprocessor.hpp"

namespace deep_object_detection
{

DeepObjectDetectionNode::DeepObjectDetectionNode(const rclcpp::NodeOptions & options)
: LifecycleNode("deep_object_detection_node", options)
{
  declareParameters();
  RCLCPP_INFO(this->get_logger(), "Deep object detection node created, waiting for configuration");
}

void DeepObjectDetectionNode::declareParameters()
{
  // Declare parameters
  this->declare_parameter<std::string>("model_path", "");
  this->declare_parameter<std::string>("class_names_path", "");
  this->declare_parameter<int>("model.num_classes", 80);
  this->declare_parameter<std::string>("model.bbox_format", "cxcywh");

  this->declare_parameter<int>("preprocessing.input_width", 640);
  this->declare_parameter<int>("preprocessing.input_height", 640);
  this->declare_parameter<std::string>("preprocessing.normalization_type", "scale_0_1");
  this->declare_parameter<std::vector<double>>("preprocessing.mean", {0.0, 0.0, 0.0});
  this->declare_parameter<std::vector<double>>("preprocessing.std", {1.0, 1.0, 1.0});
  this->declare_parameter<std::string>("preprocessing.resize_method", "letterbox");
  this->declare_parameter<int>("preprocessing.pad_value", 114);
  this->declare_parameter<std::string>("preprocessing.color_format", "rgb");

  this->declare_parameter<double>("postprocessing.score_threshold", 0.25);
  this->declare_parameter<double>("postprocessing.nms_iou_threshold", 0.45);
  this->declare_parameter<int>("postprocessing.max_detections", 300);
  this->declare_parameter<std::string>("postprocessing.score_activation", "sigmoid");
  this->declare_parameter<bool>("postprocessing.enable_nms", true);
  this->declare_parameter<bool>("postprocessing.use_multi_output", false);
  this->declare_parameter<int>("postprocessing.output_boxes_idx", 0);
  this->declare_parameter<int>("postprocessing.output_scores_idx", 1);
  this->declare_parameter<int>("postprocessing.output_classes_idx", 2);
  this->declare_parameter<std::string>("postprocessing.class_score_mode", "all_classes");
  this->declare_parameter<int>("postprocessing.class_score_start_idx", -1);
  this->declare_parameter<int>("postprocessing.class_score_count", -1);

  this->declare_parameter<bool>("postprocessing.layout.auto_detect", true);
  this->declare_parameter<int>("postprocessing.layout.batch_dim", 0);
  this->declare_parameter<int>("postprocessing.layout.detection_dim", 1);
  this->declare_parameter<int>("postprocessing.layout.feature_dim", 2);
  this->declare_parameter<int>("postprocessing.layout.bbox_start_idx", 0);
  this->declare_parameter<int>("postprocessing.layout.bbox_count", 4);
  this->declare_parameter<int>("postprocessing.layout.score_idx", 4);
  this->declare_parameter<int>("postprocessing.layout.class_idx", 5);

  // Backend plugin parameters
  this->declare_parameter<std::string>("Backend.plugin", "");
  this->declare_parameter<std::string>("Backend.execution_provider", "tensorrt");
  this->declare_parameter<int>("Backend.device_id", 0);
  this->declare_parameter<bool>("Backend.trt_engine_cache_enable", false);
  this->declare_parameter<std::string>("Backend.trt_engine_cache_path", "/tmp/deep_ros_ort_trt_cache");

  // get the parameter values, changed via topic remaps
  input_topic_ = "/multi_camera_sync/multi_image_compressed";
  output_annotations_topic_ = "/image_annotations";

  params_.model_path = this->get_parameter("model_path").as_string();
  params_.model_metadata.num_classes = this->get_parameter("model.num_classes").as_int();
  params_.model_metadata.class_names_file = this->get_parameter("class_names_path").as_string();
  {
    const std::string format = this->get_parameter("model.bbox_format").as_string();
    if (format == "xyxy") {
      params_.model_metadata.bbox_format = BboxFormat::XYXY;
    } else if (format == "xywh") {
      params_.model_metadata.bbox_format = BboxFormat::XYWH;
    } else {
      params_.model_metadata.bbox_format = BboxFormat::CXCYWH;  // default
    }
  }

  // Preprocessing parameters
  params_.preprocessing.input_width = this->get_parameter("preprocessing.input_width").as_int();
  params_.preprocessing.input_height = this->get_parameter("preprocessing.input_height").as_int();
  {
    const std::string type = this->get_parameter("preprocessing.normalization_type").as_string();
    if (type == "imagenet") {
      params_.preprocessing.normalization_type = NormalizationType::IMAGENET;
    } else if (type == "custom") {
      params_.preprocessing.normalization_type = NormalizationType::CUSTOM;
    } else if (type == "none") {
      params_.preprocessing.normalization_type = NormalizationType::NONE;
    } else {
      params_.preprocessing.normalization_type = NormalizationType::SCALE_0_1;  // default
    }
  }
  auto mean_d = this->get_parameter("preprocessing.mean").as_double_array();
  auto std_d = this->get_parameter("preprocessing.std").as_double_array();
  params_.preprocessing.mean = {
    static_cast<float>(mean_d[0]), static_cast<float>(mean_d[1]), static_cast<float>(mean_d[2])};
  params_.preprocessing.std = {
    static_cast<float>(std_d[0]), static_cast<float>(std_d[1]), static_cast<float>(std_d[2])};
  {
    const std::string method = this->get_parameter("preprocessing.resize_method").as_string();
    if (method == "resize") {
      params_.preprocessing.resize_method = ResizeMethod::RESIZE;
    } else if (method == "crop") {
      params_.preprocessing.resize_method = ResizeMethod::CROP;
    } else if (method == "pad") {
      params_.preprocessing.resize_method = ResizeMethod::PAD;
    } else {
      params_.preprocessing.resize_method = ResizeMethod::LETTERBOX;  // default
    }
  }
  params_.preprocessing.pad_value = this->get_parameter("preprocessing.pad_value").as_int();
  params_.preprocessing.color_format = this->get_parameter("preprocessing.color_format").as_string();

  // Postprocessing parameters
  params_.postprocessing.score_threshold =
    static_cast<float>(this->get_parameter("postprocessing.score_threshold").as_double());
  params_.postprocessing.nms_iou_threshold =
    static_cast<float>(this->get_parameter("postprocessing.nms_iou_threshold").as_double());
  params_.postprocessing.max_detections = this->get_parameter("postprocessing.max_detections").as_int();
  {
    const std::string activation = this->get_parameter("postprocessing.score_activation").as_string();
    if (activation == "softmax") {
      params_.postprocessing.score_activation = ScoreActivation::SOFTMAX;
    } else if (activation == "none") {
      params_.postprocessing.score_activation = ScoreActivation::NONE;
    } else {
      params_.postprocessing.score_activation = ScoreActivation::SIGMOID;  // default
    }
  }
  params_.postprocessing.enable_nms = this->get_parameter("postprocessing.enable_nms").as_bool();
  params_.postprocessing.use_multi_output = this->get_parameter("postprocessing.use_multi_output").as_bool();
  params_.postprocessing.output_boxes_idx = this->get_parameter("postprocessing.output_boxes_idx").as_int();
  params_.postprocessing.output_scores_idx = this->get_parameter("postprocessing.output_scores_idx").as_int();
  params_.postprocessing.output_classes_idx = this->get_parameter("postprocessing.output_classes_idx").as_int();
  {
    const std::string mode = this->get_parameter("postprocessing.class_score_mode").as_string();
    if (mode == "single_confidence") {
      params_.postprocessing.class_score_mode = ClassScoreMode::SINGLE_CONFIDENCE;
    } else {
      params_.postprocessing.class_score_mode = ClassScoreMode::ALL_CLASSES;  // default
    }
  }
  params_.postprocessing.class_score_start_idx = this->get_parameter("postprocessing.class_score_start_idx").as_int();
  params_.postprocessing.class_score_count = this->get_parameter("postprocessing.class_score_count").as_int();

  params_.postprocessing.layout.auto_detect = this->get_parameter("postprocessing.layout.auto_detect").as_bool();
  params_.postprocessing.layout.batch_dim = this->get_parameter("postprocessing.layout.batch_dim").as_int();
  params_.postprocessing.layout.detection_dim = this->get_parameter("postprocessing.layout.detection_dim").as_int();
  params_.postprocessing.layout.feature_dim = this->get_parameter("postprocessing.layout.feature_dim").as_int();
  params_.postprocessing.layout.bbox_start_idx = this->get_parameter("postprocessing.layout.bbox_start_idx").as_int();
  params_.postprocessing.layout.bbox_count = this->get_parameter("postprocessing.layout.bbox_count").as_int();
  params_.postprocessing.layout.score_idx = this->get_parameter("postprocessing.layout.score_idx").as_int();
  params_.postprocessing.layout.class_idx = this->get_parameter("postprocessing.layout.class_idx").as_int();

  // Backend parameters
  params_.backend.plugin = this->get_parameter("Backend.plugin").as_string();
  params_.backend.execution_provider = this->get_parameter("Backend.execution_provider").as_string();
  params_.backend.device_id = this->get_parameter("Backend.device_id").as_int();
  params_.backend.trt_engine_cache_enable = this->get_parameter("Backend.trt_engine_cache_enable").as_bool();
  params_.backend.trt_engine_cache_path = this->get_parameter("Backend.trt_engine_cache_path").as_string();
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_configure(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Configuring deep object detection node");

  try {
    loadClassNames();
    preprocessor_ = std::make_unique<ImagePreprocessor>(params_.preprocessing);

    // Load backend plugin
    if (params_.backend.plugin.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Backend.plugin parameter is required but not set");
      cleanupPartialConfiguration();
      return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
    }

    try {
      plugin_loader_ = std::make_unique<pluginlib::ClassLoader<deep_ros::DeepBackendPlugin>>(
        "deep_core", "deep_ros::DeepBackendPlugin");
      plugin_ = plugin_loader_->createUniqueInstance(params_.backend.plugin);
      plugin_->initialize(this->shared_from_this());
      allocator_ = plugin_->get_allocator();
      executor_ = plugin_->get_inference_executor();

      if (!executor_ || !allocator_) {
        throw std::runtime_error("Plugin did not provide executor or allocator");
      }

      // Load model
      if (params_.model_path.empty()) {
        throw std::runtime_error("Model path is not set");
      }

      if (!executor_->load_model(params_.model_path)) {
        throw std::runtime_error("Failed to load model: " + params_.model_path);
      }

      RCLCPP_INFO(this->get_logger(), "Loaded backend plugin: %s", plugin_->backend_name().c_str());
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Backend plugin initialization failed: %s", e.what());
      cleanupPartialConfiguration();
      return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
    }

    // dynamically get the output shape by running a dummy inference
    std::vector<size_t> input_shape = {
      1,
      RGB_CHANNELS,
      static_cast<size_t>(params_.preprocessing.input_height),
      static_cast<size_t>(params_.preprocessing.input_width)};
    std::vector<size_t> output_shape;
    try {
      PackedInput dummy;
      dummy.shape = input_shape;
      size_t total_elements = 1;
      for (size_t dim : input_shape) {
        total_elements *= dim;
      }
      dummy.data.assign(total_elements, 0.0f);

      deep_ros::Tensor input_tensor(dummy.shape, deep_ros::DataType::FLOAT32, allocator_);
      const size_t bytes = dummy.data.size() * sizeof(float);
      allocator_->copy_from_host(input_tensor.data(), dummy.data.data(), bytes);

      auto output_tensor = executor_->run_inference(input_tensor);
      output_shape = output_tensor.shape();
    } catch (const std::exception & e) {
      RCLCPP_WARN(this->get_logger(), "Could not determine output shape: %s", e.what());
      output_shape.clear();
    }

    auto formatShape = [](const std::vector<size_t> & shape) {
      if (shape.empty()) return std::string("auto-detect");
      std::string result;
      for (size_t i = 0; i < shape.size(); ++i) {
        result += std::to_string(shape[i]);
        if (i + 1 < shape.size()) result += ", ";
      }
      return result;
    };

    if (!output_shape.empty()) {
      RCLCPP_INFO(this->get_logger(), "Detected model output shape: [%s]", formatShape(output_shape).c_str());
    }

    const bool use_letterbox = (params_.preprocessing.resize_method == ResizeMethod::LETTERBOX);

    GenericPostprocessor::OutputLayout layout =
      GenericPostprocessor::autoConfigure(output_shape, params_.postprocessing.layout);
    if (layout.auto_detect && !output_shape.empty()) {
      RCLCPP_INFO(
        this->get_logger(),
        "Auto-detected layout: batch_dim=%zu, detection_dim=%zu, feature_dim=%zu",
        layout.batch_dim,
        layout.detection_dim,
        layout.feature_dim);
    } else if (!layout.auto_detect) {
      RCLCPP_INFO(
        this->get_logger(),
        "Using manual layout: batch_dim=%zu, detection_dim=%zu, feature_dim=%zu",
        layout.batch_dim,
        layout.detection_dim,
        layout.feature_dim);
    } else {
      RCLCPP_INFO(this->get_logger(), "Layout will be auto-detected from first inference");
    }

    postprocessor_ = std::make_unique<GenericPostprocessor>(
      params_.postprocessing,
      layout,
      params_.model_metadata.bbox_format,
      params_.model_metadata.num_classes,
      class_names_,
      use_letterbox);

    auto pub = this->create_publisher<Detection2DArrayMsg>(params_.output_detections_topic, rclcpp::SensorDataQoS{});
    detection_pub_ = std::static_pointer_cast<rclcpp_lifecycle::LifecyclePublisher<Detection2DArrayMsg>>(pub);

    auto marker_pub =
      this->create_publisher<visualization_msgs::msg::ImageMarker>(output_annotations_topic_, rclcpp::SensorDataQoS{});
    image_marker_pub_ =
      std::static_pointer_cast<rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::ImageMarker>>(marker_pub);

    RCLCPP_INFO(this->get_logger(), "Will subscribe to MultiImage topic: %s", input_topic_.c_str());

    RCLCPP_INFO(
      this->get_logger(),
      "Deep object detection node configured. Model: %s, input size: %dx%d, backend: %s, "
      "postprocessor: %s, output shape: [%s]",
      params_.model_path.c_str(),
      params_.preprocessing.input_width,
      params_.preprocessing.input_height,
      plugin_->backend_name().c_str(),
      postprocessor_->getFormatName().c_str(),
      formatShape(output_shape).c_str());

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to configure: %s", e.what());
    cleanupPartialConfiguration();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
  }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_activate(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Activating deep object detection node");

  // Check if backend is initialized
  if (!plugin_ || !executor_) {
    RCLCPP_ERROR(this->get_logger(), "Backend plugin not initialized - cannot activate");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
  }

  try {
    callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    // Create subscription
    setupSubscription();

    // Activate publishers
    if (detection_pub_) {
      detection_pub_->on_activate();
    }

    if (image_marker_pub_) {
      image_marker_pub_->on_activate();
    }

    RCLCPP_INFO(
      this->get_logger(), "Deep object detection node activated with backend: %s", plugin_->backend_name().c_str());
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to activate: %s", e.what());
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
  }
}

void DeepObjectDetectionNode::cleanupPartialConfiguration()
{
  postprocessor_.reset();
  executor_.reset();
  allocator_.reset();
  plugin_.reset();
  plugin_loader_.reset();
  preprocessor_.reset();
  detection_pub_.reset();
  image_marker_pub_.reset();
}

void DeepObjectDetectionNode::cleanupAllResources()
{
  stopSubscriptions();
  if (executor_) {
    executor_->unload_model();
  }
  executor_.reset();
  allocator_.reset();
  plugin_.reset();
  plugin_loader_.reset();
  preprocessor_.reset();
  postprocessor_.reset();
  class_names_.clear();
  detection_pub_.reset();
  image_marker_pub_.reset();
}

void DeepObjectDetectionNode::stopSubscriptions()
{
  if (multi_image_sub_) {
    multi_image_sub_.reset();
  }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Deactivating deep object detection node");

  // Reset subscription
  stopSubscriptions();

  // Deactivate publishers
  if (detection_pub_) {
    detection_pub_->on_deactivate();
  }

  if (image_marker_pub_) {
    image_marker_pub_->on_deactivate();
  }

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Cleaning up deep object detection node");

  // Reset all resources
  stopSubscriptions();
  detection_pub_.reset();
  image_marker_pub_.reset();
  if (executor_) {
    executor_->unload_model();
  }
  executor_.reset();
  allocator_.reset();
  plugin_.reset();
  plugin_loader_.reset();
  preprocessor_.reset();
  postprocessor_.reset();
  class_names_.clear();

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_shutdown(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Shutting down deep object detection node");

  // Reset all resources
  stopSubscriptions();
  detection_pub_.reset();
  image_marker_pub_.reset();
  if (executor_) {
    executor_->unload_model();
  }
  executor_.reset();
  allocator_.reset();
  plugin_.reset();
  plugin_loader_.reset();
  preprocessor_.reset();
  postprocessor_.reset();
  class_names_.clear();

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

void DeepObjectDetectionNode::setupSubscription()
{
  auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(1));
  qos_profile.best_effort();

  rclcpp::SubscriptionOptions options;
  options.callback_group = callback_group_;

  multi_image_sub_ = this->create_subscription<deep_msgs::msg::MultiImage>(
    input_topic_,
    qos_profile,
    [this](const deep_msgs::msg::MultiImage::ConstSharedPtr msg) {
      try {
        this->onMultiImage(msg);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in onMultiImage callback: %s", e.what());
      }
    },
    options);
  RCLCPP_INFO(this->get_logger(), "Subscribed to MultiImage topic: %s", input_topic_.c_str());
}

void DeepObjectDetectionNode::onMultiImage(const deep_msgs::msg::MultiImage::ConstSharedPtr & msg)
{
  RCLCPP_DEBUG(this->get_logger(), "Received MultiImage message with %zu images", msg->images.size());
  try {
    processMultiImage(msg);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Exception processing MultiImage: %s", e.what());
  }
}

void DeepObjectDetectionNode::processMultiImage(const deep_msgs::msg::MultiImage::ConstSharedPtr & msg)
{
  if (!executor_ || !allocator_) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Cannot process MultiImage: backend not initialized (executor: %s, allocator: %s)",
      executor_ ? "exists" : "null",
      allocator_ ? "exists" : "null");
    return;
  }

  if (msg->images.empty()) {
    RCLCPP_WARN(this->get_logger(), "Received empty MultiImage message, skipping");
    return;
  }

  auto start_time = std::chrono::steady_clock::now();
  std::vector<cv::Mat> processed;
  std::vector<ImageMeta> metas;
  std::vector<std_msgs::msg::Header> headers;
  processed.reserve(msg->images.size());
  metas.reserve(msg->images.size());
  headers.reserve(msg->images.size());

  // Decode and preprocess all images in the MultiImage message
  for (const auto & compressed_img : msg->images) {
    cv::Mat decoded = cv::imdecode(compressed_img.data, cv::IMREAD_COLOR);
    if (decoded.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to decode compressed image, skipping");
      continue;
    }

    ImageMeta meta;
    cv::Mat preprocessed = preprocessor_->preprocess(decoded, meta);
    if (preprocessed.empty()) {
      RCLCPP_WARN(this->get_logger(), "Preprocessing returned empty image, skipping");
      continue;
    }
    processed.push_back(std::move(preprocessed));
    metas.push_back(meta);
    headers.push_back(compressed_img.header);
  }

  if (processed.empty()) {
    RCLCPP_WARN(this->get_logger(), "No valid images after preprocessing, skipping MultiImage");
    return;
  }

  const auto & packed_input = preprocessor_->pack(processed);
  if (packed_input.data.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Packed input is empty after preprocessing %zu images", processed.size());
    return;
  }

  // Build input tensor
  deep_ros::Tensor input_tensor(packed_input.shape, deep_ros::DataType::FLOAT32, allocator_);
  const size_t bytes = packed_input.data.size() * sizeof(float);
  allocator_->copy_from_host(input_tensor.data(), packed_input.data.data(), bytes);

  // Run inference
  std::vector<std::vector<SimpleDetection>> batch_detections;
  if (params_.postprocessing.use_multi_output) {
    auto output_tensor = executor_->run_inference(input_tensor);
    batch_detections = postprocessor_->decodeMultiOutput({output_tensor}, metas);
  } else {
    auto output_tensor = executor_->run_inference(input_tensor);
    batch_detections = postprocessor_->decode(output_tensor, metas);
  }

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "MultiImage processing completed: %zu images in %" PRId64 " ms, total detections: %zu",
    processed.size(),
    static_cast<int64_t>(elapsed_ms),
    std::accumulate(batch_detections.begin(), batch_detections.end(), size_t(0), [](size_t sum, const auto & dets) {
      return sum + dets.size();
    }));

  publishDetections(batch_detections, headers, metas);
}

void DeepObjectDetectionNode::publishDetections(
  const std::vector<std::vector<SimpleDetection>> & batch_detections,
  const std::vector<std_msgs::msg::Header> & headers,
  const std::vector<ImageMeta> & metas)
{
  if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    return;
  }

  if (!detection_pub_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000, "Cannot publish detections: publisher not initialized");
    return;
  }

  size_t total_published = 0;
  for (size_t i = 0; i < batch_detections.size() && i < headers.size() && i < metas.size(); ++i) {
    Detection2DArrayMsg msg;
    postprocessor_->fillDetectionMessage(headers[i], batch_detections[i], metas[i], msg);
    detection_pub_->publish(msg);
    total_published += batch_detections[i].size();

    RCLCPP_DEBUG(
      this->get_logger(),
      "  Published [%zu]: frame_id=%s, %zu detections",
      i,
      headers[i].frame_id.c_str(),
      batch_detections[i].size());

    // Also publish ImageMarker annotations if publisher is available
    if (image_marker_pub_) {
      auto marker_msg = detectionsToImageMarker(headers[i], batch_detections[i]);
      image_marker_pub_->publish(marker_msg);
    }
  }
  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "Published %zu detection messages (%zu total detections)",
    batch_detections.size(),
    total_published);
}

void DeepObjectDetectionNode::loadClassNames()
{
  const auto class_names_path = this->get_parameter("class_names_path").as_string();
  if (class_names_path.empty()) {
    return;
  }

  class_names_.clear();

  std::ifstream file(class_names_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open class_names_path: " + class_names_path);
  }

  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty()) {
      class_names_.push_back(line);
    }
  }

  RCLCPP_INFO(this->get_logger(), "Loaded %zu class names from %s", class_names_.size(), class_names_path.c_str());
}

visualization_msgs::msg::ImageMarker DeepObjectDetectionNode::detectionsToImageMarker(
  const std_msgs::msg::Header & header, const std::vector<SimpleDetection> & detections) const
{
  visualization_msgs::msg::ImageMarker marker_msg;
  marker_msg.header = header;
  marker_msg.ns = "detections";
  marker_msg.id = 0;
  marker_msg.type = visualization_msgs::msg::ImageMarker::LINE_LIST;
  marker_msg.action = visualization_msgs::msg::ImageMarker::ADD;

  // Set marker lifetime
  marker_msg.lifetime.sec = 0;
  marker_msg.lifetime.nanosec = 50000000;  // 0.05 seconds - short lifetime to avoid ghosting
  marker_msg.scale = 2.0;

  // Set outline color to green
  marker_msg.outline_color.r = 0.0;
  marker_msg.outline_color.g = 1.0;
  marker_msg.outline_color.b = 0.0;
  marker_msg.outline_color.a = 1.0;

  // Iterate through detections and add bounding box line segments
  for (const auto & det : detections) {
    // Skip detections with invalid coordinates
    if (det.width <= 0 || det.height <= 0) {
      continue;
    }

    // Calculate bounding box corners
    float x_min = det.x;
    float y_min = det.y;
    float x_max = det.x + det.width;
    float y_max = det.y + det.height;

    // Add line points for bounding box rectangle (4 line segments = 8 points)
    // Top-left to top-right
    geometry_msgs::msg::Point pt1, pt2;
    pt1.x = x_min;
    pt1.y = y_min;
    pt1.z = 0;
    pt2.x = x_max;
    pt2.y = y_min;
    pt2.z = 0;
    marker_msg.points.push_back(pt1);
    marker_msg.points.push_back(pt2);

    // Top-right to bottom-right
    pt1.x = x_max;
    pt1.y = y_min;
    pt1.z = 0;
    pt2.x = x_max;
    pt2.y = y_max;
    pt2.z = 0;
    marker_msg.points.push_back(pt1);
    marker_msg.points.push_back(pt2);

    // Bottom-right to bottom-left
    pt1.x = x_max;
    pt1.y = y_max;
    pt1.z = 0;
    pt2.x = x_min;
    pt2.y = y_max;
    pt2.z = 0;
    marker_msg.points.push_back(pt1);
    marker_msg.points.push_back(pt2);

    // Bottom-left to top-left
    pt1.x = x_min;
    pt1.y = y_max;
    pt1.z = 0;
    pt2.x = x_min;
    pt2.y = y_min;
    pt2.z = 0;
    marker_msg.points.push_back(pt1);
    marker_msg.points.push_back(pt2);
  }

  return marker_msg;
}

std::shared_ptr<rclcpp_lifecycle::LifecycleNode> createDeepObjectDetectionNode(const rclcpp::NodeOptions & options)
{
  return std::make_shared<DeepObjectDetectionNode>(options);
}

}  // namespace deep_object_detection

RCLCPP_COMPONENTS_REGISTER_NODE(deep_object_detection::DeepObjectDetectionNode)
