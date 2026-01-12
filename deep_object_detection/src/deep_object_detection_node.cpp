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
#include <atomic>
#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>

#include "deep_object_detection/backend_manager.hpp"
#include "deep_object_detection/detection_types.hpp"
#include "deep_object_detection/generic_postprocessor.hpp"

#include <deep_msgs/msg/multi_image.hpp>

namespace deep_object_detection
{

DeepObjectDetectionNode::DeepObjectDetectionNode(const rclcpp::NodeOptions & options)
: LifecycleNode("deep_object_detection_node", options)
{
  declareAndReadParameters();
  RCLCPP_INFO(this->get_logger(), "Deep object detection node created, waiting for configuration");
}

void DeepObjectDetectionNode::declareAndReadParameters()
{
  params_.model_path = this->declare_parameter<std::string>("model_path", "");

  auto class_names_path = this->declare_parameter<std::string>("class_names_path", "");
  params_.model_metadata.num_classes = this->declare_parameter<int>("model.num_classes", 80);
  params_.model_metadata.class_names_file = class_names_path;
  auto bbox_format_str = this->declare_parameter<std::string>("model.bbox_format", "cxcywh");
  params_.model_metadata.bbox_format = stringToBboxFormat(bbox_format_str);

  params_.preprocessing.input_width = this->declare_parameter<int>("preprocessing.input_width", 640);
  params_.preprocessing.input_height = this->declare_parameter<int>("preprocessing.input_height", 640);
  auto normalization_type_str = this->declare_parameter<std::string>("preprocessing.normalization_type", "scale_0_1");
  params_.preprocessing.normalization_type = stringToNormalizationType(normalization_type_str);
  auto mean_d = this->declare_parameter<std::vector<double>>("preprocessing.mean", {0.0, 0.0, 0.0});
  auto std_d = this->declare_parameter<std::vector<double>>("preprocessing.std", {1.0, 1.0, 1.0});
  params_.preprocessing.mean.clear();
  params_.preprocessing.std.clear();
  for (auto v : mean_d) params_.preprocessing.mean.push_back(static_cast<float>(v));
  for (auto v : std_d) params_.preprocessing.std.push_back(static_cast<float>(v));
  auto resize_method_str = this->declare_parameter<std::string>("preprocessing.resize_method", "letterbox");
  params_.preprocessing.resize_method = stringToResizeMethod(resize_method_str);
  params_.preprocessing.pad_value = this->declare_parameter<int>("preprocessing.pad_value", 114);
  params_.preprocessing.color_format = this->declare_parameter<std::string>("preprocessing.color_format", "rgb");

  params_.postprocessing.score_threshold =
    static_cast<float>(this->declare_parameter<double>("postprocessing.score_threshold", 0.25));
  params_.postprocessing.nms_iou_threshold =
    static_cast<float>(this->declare_parameter<double>("postprocessing.nms_iou_threshold", 0.45));
  params_.postprocessing.max_detections = this->declare_parameter<int>("postprocessing.max_detections", 300);
  auto score_activation_str = this->declare_parameter<std::string>("postprocessing.score_activation", "sigmoid");
  params_.postprocessing.score_activation = stringToScoreActivation(score_activation_str);
  params_.postprocessing.enable_nms = this->declare_parameter<bool>("postprocessing.enable_nms", true);

  auto class_score_mode_str = this->declare_parameter<std::string>("postprocessing.class_score_mode", "all_classes");
  params_.postprocessing.class_score_mode = stringToClassScoreMode(class_score_mode_str);
  params_.postprocessing.class_score_start_idx =
    this->declare_parameter<int>("postprocessing.class_score_start_idx", -1);
  params_.postprocessing.class_score_count = this->declare_parameter<int>("postprocessing.class_score_count", -1);

  auto coordinate_space_str = this->declare_parameter<std::string>("postprocessing.coordinate_space", "preprocessed");
  params_.postprocessing.coordinate_space = stringToCoordinateSpace(coordinate_space_str);

  params_.postprocessing.use_multi_output = this->declare_parameter<bool>("postprocessing.use_multi_output", false);
  params_.postprocessing.output_boxes_idx = this->declare_parameter<int>("postprocessing.output_boxes_idx", 0);
  params_.postprocessing.output_scores_idx = this->declare_parameter<int>("postprocessing.output_scores_idx", 1);
  params_.postprocessing.output_classes_idx = this->declare_parameter<int>("postprocessing.output_classes_idx", 2);

  params_.postprocessing.layout.auto_detect = this->declare_parameter<bool>("postprocessing.layout.auto_detect", true);
  params_.postprocessing.layout.batch_dim = this->declare_parameter<int>("postprocessing.layout.batch_dim", 0);
  params_.postprocessing.layout.detection_dim = this->declare_parameter<int>("postprocessing.layout.detection_dim", 1);
  params_.postprocessing.layout.feature_dim = this->declare_parameter<int>("postprocessing.layout.feature_dim", 2);
  params_.postprocessing.layout.bbox_start_idx =
    this->declare_parameter<int>("postprocessing.layout.bbox_start_idx", 0);
  params_.postprocessing.layout.bbox_count = this->declare_parameter<int>("postprocessing.layout.bbox_count", 4);
  params_.postprocessing.layout.score_idx = this->declare_parameter<int>("postprocessing.layout.score_idx", 4);
  params_.postprocessing.layout.class_idx = this->declare_parameter<int>("postprocessing.layout.class_idx", 5);

  camera_sync_topic_ = this->declare_parameter<std::string>("camera_sync_topic", "");
  use_camera_sync_ = this->declare_parameter<bool>("use_camera_sync", false);
  if (!use_camera_sync_) {
    use_camera_sync_ = !camera_sync_topic_.empty();
  }
  this->declare_parameter<std::vector<std::string>>("camera_topics", std::vector<std::string>{});
  try {
    params_.camera_topics = this->get_parameter("camera_topics").as_string_array();
  } catch (const std::exception &) {
    params_.camera_topics = std::vector<std::string>{};
  }
  params_.input_qos_reliability = this->declare_parameter<std::string>("input_qos_reliability", "best_effort");
  params_.output_detections_topic = this->declare_parameter<std::string>("output_detections_topic", "/detections");

  params_.min_batch_size = this->declare_parameter<int>("min_batch_size", 1);
  params_.max_batch_size = this->declare_parameter<int>("max_batch_size", 3);
  params_.max_batch_latency_ms = this->declare_parameter<int>("max_batch_latency_ms", 0);
  params_.queue_size = this->declare_parameter<int>("queue_size", 10);
  auto queue_overflow_policy_str =
    this->declare_parameter<std::string>("queue_overflow_policy", "drop_oldest");
  params_.queue_overflow_policy = stringToQueueOverflowPolicy(queue_overflow_policy_str);
  auto decode_failure_policy_str = this->declare_parameter<std::string>("decode_failure_policy", "drop");
  params_.decode_failure_policy = stringToDecodeFailurePolicy(decode_failure_policy_str);

  params_.preferred_provider = this->declare_parameter<std::string>("preferred_provider", "tensorrt");
  params_.device_id = this->declare_parameter<int>("device_id", 0);
  params_.warmup_tensor_shapes = this->declare_parameter<bool>("warmup_tensor_shapes", true);
  params_.enable_trt_engine_cache = this->declare_parameter<bool>("enable_trt_engine_cache", false);
  params_.trt_engine_cache_path =
    this->declare_parameter<std::string>("trt_engine_cache_path", "/tmp/deep_ros_ort_trt_cache");

  this->declare_parameter("Backend.device_id", params_.device_id);
  this->declare_parameter("Backend.execution_provider", params_.preferred_provider);
  this->declare_parameter("Backend.trt_engine_cache_enable", params_.enable_trt_engine_cache);
  this->declare_parameter("Backend.trt_engine_cache_path", params_.trt_engine_cache_path);
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_configure(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Configuring deep object detection node");

  try {
    loadClassNames();
    preprocessor_ = std::make_unique<ImagePreprocessor>(params_.preprocessing);
    backend_manager_ = std::make_unique<BackendManager>(*this, params_);

    try {
      backend_manager_->initialize();
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Backend initialization failed: %s", e.what());
      return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
    }

    const size_t batch = static_cast<size_t>(params_.max_batch_size);
    const size_t channels = RGB_CHANNELS;
    const size_t height = static_cast<size_t>(params_.preprocessing.input_height);
    const size_t width = static_cast<size_t>(params_.preprocessing.input_width);
    std::vector<size_t> input_shape = {batch, channels, height, width};
    std::vector<size_t> output_shape = backend_manager_->getOutputShape(input_shape);
    if (!output_shape.empty()) {
      RCLCPP_INFO(this->get_logger(), "Detected model output shape: [%s]", formatShape(output_shape).c_str());
    }

    const auto & config = params_.postprocessing;
    const auto & model_meta = params_.model_metadata;
    const bool use_letterbox = (params_.preprocessing.resize_method == ResizeMethod::LETTERBOX);

    GenericPostprocessor::OutputLayout layout;
    if (!params_.postprocessing.layout.auto_detect) {
      layout.auto_detect = false;
      layout.batch_dim = static_cast<size_t>(params_.postprocessing.layout.batch_dim);
      layout.detection_dim = static_cast<size_t>(params_.postprocessing.layout.detection_dim);
      layout.feature_dim = static_cast<size_t>(params_.postprocessing.layout.feature_dim);
      layout.bbox_start_idx = static_cast<size_t>(params_.postprocessing.layout.bbox_start_idx);
      layout.bbox_count = static_cast<size_t>(params_.postprocessing.layout.bbox_count);
      layout.score_idx = static_cast<size_t>(params_.postprocessing.layout.score_idx);
      layout.class_idx = (params_.postprocessing.layout.class_idx >= 0)
                           ? static_cast<size_t>(params_.postprocessing.layout.class_idx)
                           : SIZE_MAX;
      if (!output_shape.empty()) {
        layout.shape = output_shape;
      }
      RCLCPP_INFO(
        this->get_logger(),
        "Using manual layout: batch_dim=%zu, detection_dim=%zu, feature_dim=%zu",
        layout.batch_dim,
        layout.detection_dim,
        layout.feature_dim);
    } else if (!output_shape.empty()) {
      layout = GenericPostprocessor::detectLayout(output_shape);
      RCLCPP_INFO(
        this->get_logger(),
        "Auto-detected layout: batch_dim=%zu, detection_dim=%zu, feature_dim=%zu",
        layout.batch_dim,
        layout.detection_dim,
        layout.feature_dim);
    } else {
      layout.auto_detect = true;
      RCLCPP_INFO(this->get_logger(), "Layout will be auto-detected from first inference");
    }
    postprocessor_ = std::make_unique<GenericPostprocessor>(
      config, layout, model_meta.bbox_format, model_meta.num_classes, params_.class_names, use_letterbox);

    detection_pub_ =
      this->create_publisher<Detection2DArrayMsg>(params_.output_detections_topic, rclcpp::SensorDataQoS{});

    if (use_camera_sync_) {
      if (camera_sync_topic_.empty()) {
        RCLCPP_ERROR(
          this->get_logger(),
          "use_camera_sync is true but camera_sync_topic is empty. "
          "Either set camera_sync_topic to a valid MultiImage topic, "
          "or set use_camera_sync: false and provide camera_topics.");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
      }
      RCLCPP_INFO(this->get_logger(), "Using camera sync mode - will subscribe to: %s", camera_sync_topic_.c_str());
    } else {
      if (params_.camera_topics.empty()) {
        RCLCPP_ERROR(
          this->get_logger(),
          "use_camera_sync is false but camera_topics is empty. "
          "Either set use_camera_sync: true and provide camera_sync_topic, "
          "or set camera_topics to a list of compressed image topics.");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
      }
      RCLCPP_INFO(
        this->get_logger(), "Using direct camera subscription mode - will subscribe to %zu camera topic(s)",
        params_.camera_topics.size());
    }

    std::string shape_str = output_shape.empty() ? "auto-detect" : formatShape(output_shape);
    RCLCPP_INFO(
      this->get_logger(),
      "Deep object detection node configured. Model: %s, input size: %dx%d, batch: [%d-%d], provider: %s, "
      "postprocessor: %s, output shape: [%s]",
      params_.model_path.c_str(),
      params_.preprocessing.input_width,
      params_.preprocessing.input_height,
      params_.min_batch_size,
      params_.max_batch_size,
      backend_manager_->activeProvider().c_str(),
      postprocessor_->getFormatName().c_str(),
      shape_str.c_str());

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to configure: %s", e.what());
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
  }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_activate(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Activating deep object detection node");

  try {
    if (use_camera_sync_) {
      setupCameraSyncSubscription();
    } else {
      setupMultiCameraSubscriptions();
    }

    batch_timer_ =
      this->create_wall_timer(std::chrono::milliseconds(5), std::bind(&DeepObjectDetectionNode::onBatchTimer, this));

    if (detection_pub_) {
      auto lifecycle_pub = std::dynamic_pointer_cast<rclcpp_lifecycle::LifecyclePublisher<Detection2DArrayMsg>>(
        detection_pub_);
      if (lifecycle_pub) {
        lifecycle_pub->on_activate();
      }
    }

    RCLCPP_INFO(this->get_logger(), "Deep object detection node activated and ready to process images");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to activate: %s", e.what());
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
  }
}

void DeepObjectDetectionNode::stopSubscriptionsAndTimer()
{
  if (batch_timer_) {
    batch_timer_->cancel();
    batch_timer_.reset();
  }

  multi_camera_subscriptions_.clear();
  if (multi_image_sub_) {
    multi_image_sub_.reset();
  }

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    image_queue_.clear();
  }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Deactivating deep object detection node");

  if (detection_pub_) {
    auto lifecycle_pub = std::dynamic_pointer_cast<rclcpp_lifecycle::LifecyclePublisher<Detection2DArrayMsg>>(
      detection_pub_);
    if (lifecycle_pub) {
      lifecycle_pub->on_deactivate();
    }
  }

  stopSubscriptionsAndTimer();

  RCLCPP_INFO(this->get_logger(), "Deep object detection node deactivated");
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Cleaning up deep object detection node");

  stopSubscriptionsAndTimer();

  backend_manager_.reset();
  preprocessor_.reset();
  postprocessor_.reset();
  params_.class_names.clear();
  detection_pub_.reset();

  RCLCPP_INFO(this->get_logger(), "Deep object detection node cleaned up");
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_shutdown(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Shutting down deep object detection node");

  stopSubscriptionsAndTimer();

  backend_manager_.reset();
  preprocessor_.reset();
  postprocessor_.reset();
  params_.class_names.clear();
  detection_pub_.reset();

  RCLCPP_INFO(this->get_logger(), "Deep object detection node shut down");
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

void DeepObjectDetectionNode::setupMultiCameraSubscriptions()
{
  multi_camera_subscriptions_.clear();
  multi_camera_subscriptions_.reserve(params_.camera_topics.size());
  auto qos_depth = static_cast<size_t>(params_.queue_size);
  auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(qos_depth));
  if (params_.input_qos_reliability == "reliable") {
    qos_profile.reliable();
  } else {
    qos_profile.best_effort();
  }
  for (size_t i = 0; i < params_.camera_topics.size(); ++i) {
    const auto & topic = params_.camera_topics[i];
    auto sub = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      topic, qos_profile, [this, i](const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
        this->handleCompressedImage(*msg, static_cast<int>(i));
      });
    multi_camera_subscriptions_.push_back(sub);
    RCLCPP_INFO(this->get_logger(), "Subscribed to compressed image topic %s (camera %zu)", topic.c_str(), i);
  }
}

void DeepObjectDetectionNode::setupCameraSyncSubscription()
{
  auto qos_depth = static_cast<size_t>(params_.queue_size);
  auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(qos_depth));
  if (params_.input_qos_reliability == "reliable") {
    qos_profile.reliable();
  } else {
    qos_profile.best_effort();
  }

  multi_image_sub_ = this->create_subscription<deep_msgs::msg::MultiImage>(
    camera_sync_topic_, qos_profile, [this](const deep_msgs::msg::MultiImage::ConstSharedPtr msg) {
      this->onMultiImage(msg);
    });
  RCLCPP_INFO(this->get_logger(), "Subscribed to camera sync MultiImage topic: %s", camera_sync_topic_.c_str());
}

void DeepObjectDetectionNode::onMultiImage(const deep_msgs::msg::MultiImage::ConstSharedPtr & msg)
{
  RCLCPP_INFO(this->get_logger(), "Received MultiImage message with %zu images", msg->images.size());
  for (size_t i = 0; i < msg->images.size(); ++i) {
    const auto & compressed_img = msg->images[i];
    handleCompressedImage(compressed_img, static_cast<int>(i));
  }
  RCLCPP_INFO(this->get_logger(), "Processed MultiImage message with %zu synchronized images", msg->images.size());
}

void DeepObjectDetectionNode::handleCompressedImage(
  const sensor_msgs::msg::CompressedImage & msg, int /* camera_id */)
{
  cv::Mat decoded = cv::imdecode(msg.data, cv::IMREAD_COLOR);
  if (decoded.empty()) {
    if (params_.decode_failure_policy == DecodeFailurePolicy::THROW) {
      throw std::runtime_error("Failed to decode compressed image");
    }
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000, "Failed to decode compressed image, dropping frame");
    return;
  }

  enqueueImage(std::move(decoded), msg.header);
  RCLCPP_DEBUG(
    this->get_logger(),
    "Enqueued compressed image (size: %zu bytes, decoded: %dx%d)",
    msg.data.size(),
    decoded.cols,
    decoded.rows);
}

void DeepObjectDetectionNode::enqueueImage(cv::Mat image, const std_msgs::msg::Header & header)
{
  size_t queue_size_after = 0;
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    const size_t limit = static_cast<size_t>(params_.queue_size);
    if (limit > 0 && image_queue_.size() >= limit) {
      switch (params_.queue_overflow_policy) {
        case QueueOverflowPolicy::THROW:
          throw std::runtime_error(
            "Image queue overflow: queue size (" + std::to_string(image_queue_.size()) + ") exceeds limit (" +
            std::to_string(limit) + "). Cannot enqueue new image.");
        case QueueOverflowPolicy::DROP_OLDEST:
          if (!image_queue_.empty()) {
            size_t queue_size_before = image_queue_.size();
            image_queue_.pop_front();
            RCLCPP_WARN_THROTTLE(
              this->get_logger(),
              *this->get_clock(),
              1000,
              "Queue overflow: dropping oldest image (queue size: %zu, limit: %zu)",
              queue_size_before,
              limit);
          }
          break;
        case QueueOverflowPolicy::DROP_NEWEST:
          RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Queue overflow: dropping new image (queue size: %zu, limit: %zu)",
            image_queue_.size(),
            limit);
          return;
      }
    }

    if (image_queue_.empty()) {
      first_image_timestamp_ = rclcpp::Time(header.stamp);
    }
    image_queue_.push_back({std::move(image), header});
    queue_size_after = image_queue_.size();
  }
  RCLCPP_DEBUG(
    this->get_logger(),
    "Enqueued image: queue size now %zu/%zu",
    queue_size_after,
    static_cast<size_t>(params_.queue_size));
}

std::string DeepObjectDetectionNode::formatShape(const std::vector<size_t> & shape) const
{
  std::string result;
  result.reserve(shape.size() * 4);
  for (size_t i = 0; i < shape.size(); ++i) {
    result += std::to_string(shape[i]);
    if (i + 1 < shape.size()) {
      result += ", ";
    }
  }
  return result;
}

void DeepObjectDetectionNode::onBatchTimer()
{
  if (processing_.exchange(true)) {
    return;
  }

  struct ProcessingGuard
  {
    std::atomic<bool> & flag;
    explicit ProcessingGuard(std::atomic<bool> & f)
    : flag(f)
    {}
    ~ProcessingGuard()
    {
      flag.store(false);
    }
  };
  ProcessingGuard guard(processing_);

  std::vector<QueuedImage> batch;
  const size_t min_batch = static_cast<size_t>(params_.min_batch_size);
  const size_t max_batch = static_cast<size_t>(params_.max_batch_size);
  size_t queue_size = 0;
  bool should_process = false;

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_size = image_queue_.size();

    if (queue_size == 0) {
      return;
    }

    if (queue_size >= min_batch) {
      should_process = true;
    }
    else if (params_.max_batch_latency_ms > 0 && !image_queue_.empty()) {
      auto now = this->now();
      auto elapsed_ms = (now - first_image_timestamp_).nanoseconds() / 1000000;
      if (elapsed_ms >= params_.max_batch_latency_ms) {
        should_process = true;
        RCLCPP_DEBUG(
          this->get_logger(),
          "Batch timeout exceeded (%ld ms >= %d ms), processing %zu images (min: %zu)",
          elapsed_ms,
          params_.max_batch_latency_ms,
          queue_size,
          min_batch);
      }
    }

    if (!should_process) {
      RCLCPP_DEBUG_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        5000,
        "Waiting for more images: %zu/%zu in queue (min: %zu)",
        queue_size,
        min_batch,
        min_batch);
      return;
    }

    const size_t batch_size = std::min(max_batch, queue_size);
    batch.clear();
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
      batch.push_back(std::move(image_queue_.front()));
      image_queue_.pop_front();
    }

    if (!image_queue_.empty()) {
      first_image_timestamp_ = rclcpp::Time(image_queue_.front().header.stamp);
    }
  }

  if (!batch.empty()) {
    RCLCPP_INFO(
      this->get_logger(), "Processing batch of %zu images (queue had %zu images)", batch.size(), queue_size);
    processBatch(batch);
  }
}

void DeepObjectDetectionNode::processBatch(const std::vector<QueuedImage> & batch)
{
  if (!backend_manager_ || !backend_manager_->hasExecutor()) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Cannot process batch: backend not initialized (backend_manager_: %s, hasExecutor: %s)",
      backend_manager_ ? "exists" : "null",
      (backend_manager_ && backend_manager_->hasExecutor()) ? "true" : "false");
    return;
  }
  std::vector<cv::Mat> processed;
  std::vector<ImageMeta> metas;
  std::vector<std_msgs::msg::Header> headers;
  processed.reserve(batch.size());
  metas.reserve(batch.size());
  headers.reserve(batch.size());

  for (const auto & item : batch) {
    ImageMeta meta;
    cv::Mat preprocessed = preprocessor_->preprocess(item.bgr, meta);
    if (preprocessed.empty()) {
      RCLCPP_WARN(this->get_logger(), "Preprocessing returned empty image, skipping");
      continue;
    }
    processed.push_back(std::move(preprocessed));
    metas.push_back(meta);
    headers.push_back(item.header);
  }

  if (processed.empty()) {
    RCLCPP_WARN(this->get_logger(), "No valid images after preprocessing, skipping batch");
    return;
  }

  const auto & packed_input = preprocessor_->pack(processed);
  if (packed_input.data.empty()) {
    RCLCPP_ERROR(
      this->get_logger(), "Packed input is empty after preprocessing %zu images", processed.size());
    return;
  }

  std::vector<std::vector<SimpleDetection>> batch_detections;
  if (params_.postprocessing.use_multi_output) {
    auto output_tensors = backend_manager_->inferAllOutputs(packed_input);
    batch_detections = postprocessor_->decodeMultiOutput(output_tensors, metas);
  } else {
    auto output_tensor = backend_manager_->infer(packed_input);
    batch_detections = postprocessor_->decode(output_tensor, metas);
  }
  publishDetections(batch_detections, headers, metas);
}

void DeepObjectDetectionNode::publishDetections(
  const std::vector<std::vector<SimpleDetection>> & batch_detections,
  const std::vector<std_msgs::msg::Header> & headers,
  const std::vector<ImageMeta> & metas)
{
  for (size_t i = 0; i < batch_detections.size() && i < headers.size() && i < metas.size(); ++i) {
    Detection2DArrayMsg msg;
    postprocessor_->fillDetectionMessage(headers[i], batch_detections[i], metas[i], msg);
    detection_pub_->publish(msg);
  }
}

void DeepObjectDetectionNode::loadClassNames()
{
  const auto & class_names_path = params_.model_metadata.class_names_file;
  if (class_names_path.empty()) {
    return;
  }

  params_.class_names.clear();

  std::ifstream file(class_names_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open class_names_path: " + class_names_path);
  }

  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty()) {
      params_.class_names.push_back(line);
    }
  }

  RCLCPP_INFO(
    this->get_logger(), "Loaded %zu class names from %s", params_.class_names.size(), class_names_path.c_str());
}

std::shared_ptr<rclcpp_lifecycle::LifecycleNode> createDeepObjectDetectionNode(const rclcpp::NodeOptions & options)
{
  return std::make_shared<DeepObjectDetectionNode>(options);
}

}  // namespace deep_object_detection

RCLCPP_COMPONENTS_REGISTER_NODE(deep_object_detection::DeepObjectDetectionNode)
