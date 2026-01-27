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

#include <cv_bridge/cv_bridge.h>

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

#include <deep_core/deep_node_base.hpp>
#include <deep_core/types/tensor.hpp>
#include <deep_msgs/msg/multi_image.hpp>
#include <deep_msgs/msg/multi_image_compressed.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "deep_object_detection/detection_types.hpp"
#include "deep_object_detection/generic_postprocessor.hpp"

namespace deep_object_detection
{

DeepObjectDetectionNode::DeepObjectDetectionNode(const rclcpp::NodeOptions & options)
: DeepNodeBase("deep_object_detection_node", options)
{
  declareParameters();
  RCLCPP_INFO(this->get_logger(), "Deep object detection node created, waiting for configuration");
}

void DeepObjectDetectionNode::declareParameters()
{
  // Declare parameters
  // Note: model_path and Backend.plugin are declared by DeepNodeBase
  // All parameter names use capitalized dot notation (Model.*, Preprocessing.*, Postprocessing.*, Backend.*)
  this->declare_parameter<std::string>("class_names_path", "");
  this->declare_parameter<int>("Model.num_classes", 80);
  this->declare_parameter<std::string>("Model.bbox_format", "cxcywh");

  this->declare_parameter<int>("Preprocessing.input_width", 640);
  this->declare_parameter<int>("Preprocessing.input_height", 640);
  this->declare_parameter<std::string>("Preprocessing.normalization_type", "scale_0_1");
  this->declare_parameter<std::vector<double>>("Preprocessing.mean", {0.0, 0.0, 0.0});
  this->declare_parameter<std::vector<double>>("Preprocessing.std", {1.0, 1.0, 1.0});
  this->declare_parameter<std::string>("Preprocessing.resize_method", "letterbox");
  this->declare_parameter<int>("Preprocessing.pad_value", 114);
  this->declare_parameter<std::string>("Preprocessing.color_format", "rgb");

  this->declare_parameter<double>("Postprocessing.score_threshold", 0.25);
  this->declare_parameter<double>("Postprocessing.nms_iou_threshold", 0.45);
  this->declare_parameter<std::string>("Postprocessing.score_activation", "sigmoid");
  this->declare_parameter<bool>("Postprocessing.enable_nms", true);
  this->declare_parameter<bool>("Postprocessing.use_multi_output", false);
  this->declare_parameter<int>("Postprocessing.output_boxes_idx", 0);
  this->declare_parameter<int>("Postprocessing.output_scores_idx", 1);
  this->declare_parameter<int>("Postprocessing.output_classes_idx", 2);
  this->declare_parameter<std::string>("Postprocessing.class_score_mode", "all_classes");
  this->declare_parameter<int>("Postprocessing.class_score_start_idx", -1);
  this->declare_parameter<int>("Postprocessing.class_score_count", -1);

  this->declare_parameter<bool>("Postprocessing.layout.auto_detect", true);
  this->declare_parameter<int>("Postprocessing.layout.batch_dim", 0);
  this->declare_parameter<int>("Postprocessing.layout.detection_dim", 1);
  this->declare_parameter<int>("Postprocessing.layout.feature_dim", 2);
  this->declare_parameter<int>("Postprocessing.layout.bbox_start_idx", 0);
  this->declare_parameter<int>("Postprocessing.layout.bbox_count", 4);
  this->declare_parameter<int>("Postprocessing.layout.score_idx", 4);
  this->declare_parameter<int>("Postprocessing.layout.class_idx", 5);

  // Backend plugin parameters (Backend.plugin is declared by DeepNodeBase)
  this->declare_parameter<std::string>("Backend.execution_provider", "tensorrt");
  this->declare_parameter<int>("Backend.device_id", 0);
  this->declare_parameter<bool>("Backend.trt_engine_cache_enable", false);
  this->declare_parameter<std::string>("Backend.trt_engine_cache_path", "/tmp/deep_ros_ort_trt_cache");

  // Input configuration
  this->declare_parameter<bool>("use_compressed_images", true);

  // Output configuration
  this->declare_parameter<std::string>("output_detections_topic", "/detections");

  // get the parameter values, changed via topic remaps
  input_topic_ = "/multi_camera_sync/multi_image_compressed";
  input_topic_raw_ = "/multi_camera_sync/multi_image_raw";
  output_annotations_topic_ = "/image_annotations";
  params_.output_detections_topic = this->get_parameter("output_detections_topic").as_string();

  params_.model_path = this->get_parameter("model_path").as_string();
  params_.model_metadata.num_classes = this->get_parameter("Model.num_classes").as_int();
  params_.model_metadata.class_names_file = this->get_parameter("class_names_path").as_string();
  params_.model_metadata.bbox_format = this->get_parameter("Model.bbox_format").as_string();

  // Preprocessing parameters
  params_.preprocessing.input_width = this->get_parameter("Preprocessing.input_width").as_int();
  params_.preprocessing.input_height = this->get_parameter("Preprocessing.input_height").as_int();
  params_.preprocessing.normalization_type = this->get_parameter("Preprocessing.normalization_type").as_string();
  auto mean_d = this->get_parameter("Preprocessing.mean").as_double_array();
  auto std_d = this->get_parameter("Preprocessing.std").as_double_array();
  params_.preprocessing.mean = {
    static_cast<float>(mean_d[0]), static_cast<float>(mean_d[1]), static_cast<float>(mean_d[2])};
  params_.preprocessing.std = {
    static_cast<float>(std_d[0]), static_cast<float>(std_d[1]), static_cast<float>(std_d[2])};
  params_.preprocessing.resize_method = this->get_parameter("Preprocessing.resize_method").as_string();
  params_.preprocessing.pad_value = this->get_parameter("Preprocessing.pad_value").as_int();
  params_.preprocessing.color_format = this->get_parameter("Preprocessing.color_format").as_string();

  // Postprocessing parameters
  params_.postprocessing.score_threshold =
    static_cast<float>(this->get_parameter("Postprocessing.score_threshold").as_double());
  params_.postprocessing.nms_iou_threshold =
    static_cast<float>(this->get_parameter("Postprocessing.nms_iou_threshold").as_double());
  params_.postprocessing.score_activation = this->get_parameter("Postprocessing.score_activation").as_string();
  params_.postprocessing.enable_nms = this->get_parameter("Postprocessing.enable_nms").as_bool();
  params_.postprocessing.use_multi_output = this->get_parameter("Postprocessing.use_multi_output").as_bool();
  params_.postprocessing.output_boxes_idx = this->get_parameter("Postprocessing.output_boxes_idx").as_int();
  params_.postprocessing.output_scores_idx = this->get_parameter("Postprocessing.output_scores_idx").as_int();
  params_.postprocessing.output_classes_idx = this->get_parameter("Postprocessing.output_classes_idx").as_int();
  params_.postprocessing.class_score_mode = this->get_parameter("Postprocessing.class_score_mode").as_string();
  params_.postprocessing.class_score_start_idx = this->get_parameter("Postprocessing.class_score_start_idx").as_int();
  params_.postprocessing.class_score_count = this->get_parameter("Postprocessing.class_score_count").as_int();

  params_.postprocessing.layout.auto_detect = this->get_parameter("Postprocessing.layout.auto_detect").as_bool();
  params_.postprocessing.layout.batch_dim = this->get_parameter("Postprocessing.layout.batch_dim").as_int();
  params_.postprocessing.layout.detection_dim = this->get_parameter("Postprocessing.layout.detection_dim").as_int();
  params_.postprocessing.layout.feature_dim = this->get_parameter("Postprocessing.layout.feature_dim").as_int();
  params_.postprocessing.layout.bbox_start_idx = this->get_parameter("Postprocessing.layout.bbox_start_idx").as_int();
  params_.postprocessing.layout.bbox_count = this->get_parameter("Postprocessing.layout.bbox_count").as_int();
  params_.postprocessing.layout.score_idx = this->get_parameter("Postprocessing.layout.score_idx").as_int();
  params_.postprocessing.layout.class_idx = this->get_parameter("Postprocessing.layout.class_idx").as_int();

  // Backend parameters
  params_.backend.plugin = this->get_parameter("Backend.plugin").as_string();
  params_.backend.execution_provider = this->get_parameter("Backend.execution_provider").as_string();
  params_.backend.device_id = this->get_parameter("Backend.device_id").as_int();
  params_.backend.trt_engine_cache_enable = this->get_parameter("Backend.trt_engine_cache_enable").as_bool();
  params_.backend.trt_engine_cache_path = this->get_parameter("Backend.trt_engine_cache_path").as_string();

  use_compressed_images_ = this->get_parameter("use_compressed_images").as_bool();
}

deep_ros::CallbackReturn DeepObjectDetectionNode::on_configure_impl(const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Configuring deep object detection node");

  try {
    // Check if plugin and model are loaded (handled by DeepNodeBase)
    if (!is_plugin_loaded()) {
      RCLCPP_ERROR(this->get_logger(), "Backend plugin not loaded");
      return deep_ros::CallbackReturn::FAILURE;
    }

    if (!is_model_loaded()) {
      RCLCPP_ERROR(this->get_logger(), "Model not loaded");
      return deep_ros::CallbackReturn::FAILURE;
    }

    loadClassNames();
    preprocessor_ = std::make_unique<ImagePreprocessor>(params_.preprocessing);

    // Get allocator from base class
    auto allocator = get_current_allocator();
    if (!allocator) {
      RCLCPP_ERROR(this->get_logger(), "Plugin did not provide allocator");
      return deep_ros::CallbackReturn::FAILURE;
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

      deep_ros::Tensor input_tensor(dummy.shape, deep_ros::DataType::FLOAT32, allocator);
      const size_t bytes = dummy.data.size() * sizeof(float);
      allocator->copy_from_host(input_tensor.data(), dummy.data.data(), bytes);

      auto output_tensor = run_inference(input_tensor);
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

    const bool use_letterbox = (params_.preprocessing.resize_method == "letterbox");

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
      "output shape: [%s]",
      params_.model_path.c_str(),
      params_.preprocessing.input_width,
      params_.preprocessing.input_height,
      get_backend_name().c_str(),
      formatShape(output_shape).c_str());

    return deep_ros::CallbackReturn::SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to configure: %s", e.what());
    return deep_ros::CallbackReturn::FAILURE;
  }
}

deep_ros::CallbackReturn DeepObjectDetectionNode::on_activate_impl(const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Activating deep object detection node");

  // Check if backend is initialized
  if (!is_plugin_loaded() || !is_model_loaded()) {
    RCLCPP_ERROR(this->get_logger(), "Backend plugin or model not loaded - cannot activate");
    return deep_ros::CallbackReturn::FAILURE;
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
      this->get_logger(), "Deep object detection node activated with backend: %s", get_backend_name().c_str());
    return deep_ros::CallbackReturn::SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to activate: %s", e.what());
    return deep_ros::CallbackReturn::FAILURE;
  }
}

void DeepObjectDetectionNode::stopSubscriptions()
{
  if (multi_image_sub_) {
    multi_image_sub_.reset();
  }
  if (multi_image_raw_sub_) {
    multi_image_raw_sub_.reset();
  }
}

deep_ros::CallbackReturn DeepObjectDetectionNode::on_deactivate_impl(const rclcpp_lifecycle::State & /*state*/)
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

  return deep_ros::CallbackReturn::SUCCESS;
}

deep_ros::CallbackReturn DeepObjectDetectionNode::on_cleanup_impl(const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Cleaning up deep object detection node");

  // Reset all resources
  stopSubscriptions();
  detection_pub_.reset();
  image_marker_pub_.reset();
  preprocessor_.reset();
  postprocessor_.reset();
  class_names_.clear();

  return deep_ros::CallbackReturn::SUCCESS;
}

deep_ros::CallbackReturn DeepObjectDetectionNode::on_shutdown_impl(const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Shutting down deep object detection node");

  // Reset all resources
  stopSubscriptions();
  detection_pub_.reset();
  image_marker_pub_.reset();
  preprocessor_.reset();
  postprocessor_.reset();
  class_names_.clear();

  return deep_ros::CallbackReturn::SUCCESS;
}

void DeepObjectDetectionNode::setupSubscription()
{
  auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(1));
  qos_profile.best_effort();

  rclcpp::SubscriptionOptions options;
  options.callback_group = callback_group_;

  if (use_compressed_images_) {
    // Subscribe to compressed MultiImageCompressed
    multi_image_sub_ = this->create_subscription<deep_msgs::msg::MultiImageCompressed>(
      input_topic_,
      qos_profile,
      [this](const deep_msgs::msg::MultiImageCompressed::ConstSharedPtr msg) {
        try {
          this->onMultiImage(msg);
        } catch (const std::exception & e) {
          RCLCPP_ERROR(this->get_logger(), "Exception in onMultiImage callback: %s", e.what());
        }
      },
      options);
    RCLCPP_INFO(this->get_logger(), "Subscribed to MultiImageCompressed (compressed) topic: %s", input_topic_.c_str());
  } else {
    // Subscribe to uncompressed MultiImage
    multi_image_raw_sub_ = this->create_subscription<deep_msgs::msg::MultiImage>(
      input_topic_raw_,
      qos_profile,
      [this](const deep_msgs::msg::MultiImage::ConstSharedPtr msg) {
        try {
          this->onMultiImageRaw(msg);
        } catch (const std::exception & e) {
          RCLCPP_ERROR(this->get_logger(), "Exception in onMultiImageRaw callback: %s", e.what());
        }
      },
      options);
    RCLCPP_INFO(this->get_logger(), "Subscribed to MultiImage (uncompressed) topic: %s", input_topic_raw_.c_str());
  }
}

cv::Mat DeepObjectDetectionNode::decodeCompressedImage(const sensor_msgs::msg::CompressedImage & compressed_img)
{
  return cv::imdecode(compressed_img.data, cv::IMREAD_COLOR);
}

cv::Mat DeepObjectDetectionNode::decodeImage(const sensor_msgs::msg::Image & img)
{
  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
    return cv_ptr->image;
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "cv_bridge exception: %s", e.what());
    return cv::Mat();
  }
}

void DeepObjectDetectionNode::onMultiImage(const deep_msgs::msg::MultiImageCompressed::ConstSharedPtr & msg)
{
  RCLCPP_DEBUG(this->get_logger(), "Received MultiImage message with %zu images", msg->images.size());
  try {
    std::vector<cv::Mat> images;
    std::vector<std_msgs::msg::Header> headers;
    images.reserve(msg->images.size());
    headers.reserve(msg->images.size());

    for (const auto & compressed_img : msg->images) {
      cv::Mat decoded = decodeCompressedImage(compressed_img);
      if (decoded.empty()) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000, "Failed to decode compressed image, skipping");
        continue;
      }
      images.push_back(std::move(decoded));
      headers.push_back(compressed_img.header);
    }

    if (!images.empty()) {
      processImages(images, headers);
    } else {
      RCLCPP_WARN(this->get_logger(), "No valid images after decoding, skipping MultiImage");
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Exception processing MultiImage: %s", e.what());
  }
}

void DeepObjectDetectionNode::onMultiImageRaw(const deep_msgs::msg::MultiImage::ConstSharedPtr & msg)
{
  RCLCPP_DEBUG(this->get_logger(), "Received MultiImage message with %zu images", msg->images.size());
  try {
    std::vector<cv::Mat> images;
    std::vector<std_msgs::msg::Header> headers;
    images.reserve(msg->images.size());
    headers.reserve(msg->images.size());

    for (const auto & img : msg->images) {
      cv::Mat decoded = decodeImage(img);
      if (decoded.empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to decode image, skipping");
        continue;
      }
      images.push_back(std::move(decoded));
      headers.push_back(img.header);
    }

    if (!images.empty()) {
      processImages(images, headers);
    } else {
      RCLCPP_WARN(this->get_logger(), "No valid images after decoding, skipping MultiImage");
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Exception processing MultiImage: %s", e.what());
  }
}

void DeepObjectDetectionNode::processImages(
  const std::vector<cv::Mat> & images, const std::vector<std_msgs::msg::Header> & headers)
{
  if (!is_plugin_loaded() || !is_model_loaded()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot process images: backend not initialized");
    return;
  }

  auto allocator = get_current_allocator();
  if (!allocator) {
    RCLCPP_ERROR(this->get_logger(), "Cannot process images: allocator not available");
    return;
  }

  if (images.empty()) {
    RCLCPP_WARN(this->get_logger(), "Received empty image vector, skipping");
    return;
  }

  auto start_time = std::chrono::steady_clock::now();
  std::vector<cv::Mat> processed;
  std::vector<ImageMeta> metas;
  processed.reserve(images.size());
  metas.reserve(images.size());

  // Preprocess all images
  for (const auto & img : images) {
    if (img.empty()) {
      RCLCPP_WARN(this->get_logger(), "Received empty image, skipping");
      continue;
    }

    ImageMeta meta;
    cv::Mat preprocessed = preprocessor_->preprocess(img, meta);
    if (preprocessed.empty()) {
      RCLCPP_WARN(this->get_logger(), "Preprocessing returned empty image, skipping");
      continue;
    }
    processed.push_back(std::move(preprocessed));
    metas.push_back(meta);
  }

  if (processed.empty()) {
    RCLCPP_WARN(this->get_logger(), "No valid images after preprocessing, skipping");
    return;
  }

  const auto & packed_input = preprocessor_->pack(processed);
  if (packed_input.data.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Packed input is empty after preprocessing %zu images", processed.size());
    return;
  }

  // Build input tensor
  deep_ros::Tensor input_tensor(packed_input.shape, deep_ros::DataType::FLOAT32, allocator);
  const size_t bytes = packed_input.data.size() * sizeof(float);
  allocator->copy_from_host(input_tensor.data(), packed_input.data.data(), bytes);

  // Run inference
  std::vector<std::vector<SimpleDetection>> batch_detections;
  if (params_.postprocessing.use_multi_output) {
    auto output_tensor = run_inference(input_tensor);
    batch_detections = postprocessor_->decodeMultiOutput({output_tensor}, metas);
  } else {
    auto output_tensor = run_inference(input_tensor);
    batch_detections = postprocessor_->decode(output_tensor, metas);
  }

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "Image processing completed: %zu images in %" PRId64 " ms, total detections: %zu",
    processed.size(),
    static_cast<int64_t>(elapsed_ms),
    std::accumulate(batch_detections.begin(), batch_detections.end(), size_t(0), [](size_t sum, const auto & dets) {
      return sum + dets.size();
    }));

  // Use headers that match the processed images (may be fewer if some were skipped)
  std::vector<std_msgs::msg::Header> processed_headers;
  processed_headers.reserve(processed.size());
  for (size_t i = 0; i < processed.size() && i < headers.size(); ++i) {
    processed_headers.push_back(headers[i]);
  }

  publishDetections(batch_detections, processed_headers, metas);
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

std::shared_ptr<deep_ros::DeepNodeBase> createDeepObjectDetectionNode(const rclcpp::NodeOptions & options)
{
  return std::make_shared<DeepObjectDetectionNode>(options);
}

}  // namespace deep_object_detection

RCLCPP_COMPONENTS_REGISTER_NODE(deep_object_detection::DeepObjectDetectionNode)
