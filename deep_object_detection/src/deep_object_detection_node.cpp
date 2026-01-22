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
#include <deque>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <deep_msgs/msg/multi_image.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>

#include "deep_object_detection/backend_manager.hpp"
#include "deep_object_detection/detection_types.hpp"
#include "deep_object_detection/generic_postprocessor.hpp"

namespace deep_object_detection
{

DeepObjectDetectionNode::DeepObjectDetectionNode(const rclcpp::NodeOptions & options)
: LifecycleNode("deep_object_detection_node", options)
, dropped_images_count_(0)
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
  params_.preprocessing.mean.reserve(mean_d.size());
  params_.preprocessing.std.reserve(std_d.size());
  std::transform(mean_d.begin(), mean_d.end(), std::back_inserter(params_.preprocessing.mean), [](double v) {
    return static_cast<float>(v);
  });
  std::transform(std_d.begin(), std_d.end(), std::back_inserter(params_.preprocessing.std), [](double v) {
    return static_cast<float>(v);
  });
  auto resize_method_str = this->declare_parameter<std::string>("preprocessing.resize_method", "letterbox");
  params_.preprocessing.resize_method = stringToResizeMethod(resize_method_str);
  params_.preprocessing.pad_value = 114;
  params_.preprocessing.color_format = this->declare_parameter<std::string>("preprocessing.color_format", "rgb");

  params_.postprocessing.score_threshold =
    static_cast<float>(this->declare_parameter<double>("postprocessing.score_threshold", 0.25));
  params_.postprocessing.nms_iou_threshold =
    static_cast<float>(this->declare_parameter<double>("postprocessing.nms_iou_threshold", 0.45));
  params_.postprocessing.max_detections = this->declare_parameter<int>("postprocessing.max_detections", 300);
  auto score_activation_str = this->declare_parameter<std::string>("postprocessing.score_activation", "sigmoid");
  params_.postprocessing.score_activation = stringToScoreActivation(score_activation_str);
  params_.postprocessing.enable_nms = true;

  auto class_score_mode_str = this->declare_parameter<std::string>("postprocessing.class_score_mode", "all_classes");
  params_.postprocessing.class_score_mode = stringToClassScoreMode(class_score_mode_str);
  params_.postprocessing.class_score_start_idx = -1;
  params_.postprocessing.class_score_count = -1;
  params_.postprocessing.coordinate_space = CoordinateSpace::PREPROCESSED;

  params_.postprocessing.use_multi_output = this->declare_parameter<bool>("postprocessing.use_multi_output", false);
  if (params_.postprocessing.use_multi_output) {
    params_.postprocessing.output_boxes_idx = this->declare_parameter<int>("postprocessing.output_boxes_idx", 0);
    params_.postprocessing.output_scores_idx = this->declare_parameter<int>("postprocessing.output_scores_idx", 1);
    params_.postprocessing.output_classes_idx = this->declare_parameter<int>("postprocessing.output_classes_idx", 2);
  } else {
    params_.postprocessing.output_boxes_idx = 0;
    params_.postprocessing.output_scores_idx = 1;
    params_.postprocessing.output_classes_idx = 2;
  }

  params_.postprocessing.layout.auto_detect = this->declare_parameter<bool>("postprocessing.layout.auto_detect", true);
  if (!params_.postprocessing.layout.auto_detect) {
    params_.postprocessing.layout.batch_dim = this->declare_parameter<int>("postprocessing.layout.batch_dim", 0);
    params_.postprocessing.layout.detection_dim =
      this->declare_parameter<int>("postprocessing.layout.detection_dim", 1);
    params_.postprocessing.layout.feature_dim = this->declare_parameter<int>("postprocessing.layout.feature_dim", 2);
    params_.postprocessing.layout.bbox_start_idx =
      this->declare_parameter<int>("postprocessing.layout.bbox_start_idx", 0);
    params_.postprocessing.layout.bbox_count = this->declare_parameter<int>("postprocessing.layout.bbox_count", 4);
    params_.postprocessing.layout.score_idx = this->declare_parameter<int>("postprocessing.layout.score_idx", 4);
    params_.postprocessing.layout.class_idx = this->declare_parameter<int>("postprocessing.layout.class_idx", 5);
  } else {
    params_.postprocessing.layout.batch_dim = 0;
    params_.postprocessing.layout.detection_dim = 1;
    params_.postprocessing.layout.feature_dim = 2;
    params_.postprocessing.layout.bbox_start_idx = 0;
    params_.postprocessing.layout.bbox_count = 4;
    params_.postprocessing.layout.score_idx = 4;
    params_.postprocessing.layout.class_idx = 5;
  }

  input_topic_ = this->declare_parameter<std::string>("input_topic", "");
  params_.input_qos_reliability = "best_effort";
  params_.output_detections_topic = this->declare_parameter<std::string>("output_detections_topic", "/detections");
  output_annotations_topic_ = this->declare_parameter<std::string>("output_annotations_topic", "/image_annotations");

  params_.max_batch_size = this->declare_parameter<int>("max_batch_size", 3);
  params_.queue_size = this->declare_parameter<int>("queue_size", 10);
  if (params_.queue_size < 0) {
    RCLCPP_WARN(
      this->get_logger(),
      "queue_size (%d) is negative, clamping to 0 (unlimited internal queue). "
      "Note: ROS QoS depth will use minimum of 1.",
      params_.queue_size);
    params_.queue_size = 0;
  }
  params_.queue_overflow_policy = QueueOverflowPolicy::DROP_OLDEST;
  params_.decode_failure_policy = DecodeFailurePolicy::DROP;

  params_.preferred_provider = this->declare_parameter<std::string>("preferred_provider", "tensorrt");
  params_.device_id = this->declare_parameter<int>("device_id", 0);
  params_.warmup_tensor_shapes = true;
  params_.enable_trt_engine_cache = this->declare_parameter<bool>("enable_trt_engine_cache", false);
  params_.trt_engine_cache_path =
    this->declare_parameter<std::string>("trt_engine_cache_path", "/tmp/deep_ros_ort_trt_cache");
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
      cleanupPartialConfiguration();
      return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
    }

    const size_t batch = static_cast<size_t>(params_.max_batch_size);
    const size_t channels = RGB_CHANNELS;
    const size_t height = static_cast<size_t>(params_.preprocessing.input_height);
    const size_t width = static_cast<size_t>(params_.preprocessing.input_width);
    std::vector<size_t> input_shape = {batch, channels, height, width};
    std::vector<size_t> output_shape = backend_manager_->getOutputShape(input_shape);

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

    const auto & config = params_.postprocessing;
    const auto & model_meta = params_.model_metadata;
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
      config, layout, model_meta.bbox_format, model_meta.num_classes, params_.class_names, use_letterbox);

    auto pub = this->create_publisher<Detection2DArrayMsg>(params_.output_detections_topic, rclcpp::SensorDataQoS{});
    detection_pub_ = std::static_pointer_cast<rclcpp_lifecycle::LifecyclePublisher<Detection2DArrayMsg>>(pub);

    auto marker_pub =
      this->create_publisher<visualization_msgs::msg::ImageMarker>(output_annotations_topic_, rclcpp::SensorDataQoS{});
    image_marker_pub_ =
      std::static_pointer_cast<rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::ImageMarker>>(marker_pub);

    if (input_topic_.empty()) {
      RCLCPP_ERROR(this->get_logger(), "input_topic is empty. Please set input_topic to a valid MultiImage topic.");
      cleanupPartialConfiguration();
      return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
    }
    RCLCPP_INFO(this->get_logger(), "Will subscribe to MultiImage topic: %s", input_topic_.c_str());

    RCLCPP_INFO(
      this->get_logger(),
      "Deep object detection node configured. Model: %s, input size: %dx%d, batch_size: %d, provider: %s, "
      "postprocessor: %s, output shape: [%s]",
      params_.model_path.c_str(),
      params_.preprocessing.input_width,
      params_.preprocessing.input_height,
      params_.max_batch_size,
      backend_manager_->activeProvider().c_str(),
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

  try {
    callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    setupSubscription();

    batch_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(5), std::bind(&DeepObjectDetectionNode::onBatchTimer, this), callback_group_);

    if (detection_pub_) {
      detection_pub_->on_activate();
    }

    if (image_marker_pub_) {
      image_marker_pub_->on_activate();
    }

    RCLCPP_INFO(this->get_logger(), "Deep object detection node activated and ready to process images");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to activate: %s", e.what());
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
  }
}

void DeepObjectDetectionNode::cleanupPartialConfiguration()
{
  postprocessor_.reset();
  backend_manager_.reset();
  preprocessor_.reset();
  detection_pub_.reset();
  image_marker_pub_.reset();
}

void DeepObjectDetectionNode::cleanupAllResources()
{
  stopSubscriptionsAndTimer();
  backend_manager_.reset();
  preprocessor_.reset();
  postprocessor_.reset();
  params_.class_names.clear();
  detection_pub_.reset();
  image_marker_pub_.reset();
}

void DeepObjectDetectionNode::stopSubscriptionsAndTimer()
{
  if (batch_timer_) {
    batch_timer_->cancel();
    batch_timer_.reset();
  }

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
  stopSubscriptionsAndTimer();
  if (detection_pub_) {
    detection_pub_->on_deactivate();
  }

  if (image_marker_pub_) {
    image_marker_pub_->on_deactivate();
  }

  RCLCPP_INFO(this->get_logger(), "Deep object detection node deactivated");
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Cleaning up deep object detection node");
  cleanupAllResources();
  RCLCPP_INFO(this->get_logger(), "Deep object detection node cleaned up");
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_shutdown(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Shutting down deep object detection node");
  cleanupAllResources();
  RCLCPP_INFO(this->get_logger(), "Deep object detection node shut down");
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

void DeepObjectDetectionNode::setupSubscription()
{
  const size_t qos_depth = std::max(static_cast<size_t>(1), static_cast<size_t>(params_.queue_size));
  auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(qos_depth));
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
  for (size_t i = 0; i < msg->images.size(); ++i) {
    const auto & compressed_img = msg->images[i];
    try {
      handleCompressedImage(compressed_img);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Exception processing image %zu in MultiImage: %s", i, e.what());
    }
  }
  RCLCPP_DEBUG(this->get_logger(), "Processed MultiImage message with %zu synchronized images", msg->images.size());
}

void DeepObjectDetectionNode::handleCompressedImage(const sensor_msgs::msg::CompressedImage & msg)
{
  cv::Mat decoded = cv::imdecode(msg.data, cv::IMREAD_COLOR);
  if (decoded.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000, "Failed to decode compressed image, dropping frame");
    return;
  }

  const int width = decoded.cols;
  const int height = decoded.rows;
  enqueueImage(std::move(decoded), msg.header);
  RCLCPP_DEBUG(
    this->get_logger(), "Enqueued decoded image (size: %zu bytes, decoded: %dx%d)", msg.data.size(), width, height);
}

void DeepObjectDetectionNode::enqueueImage(cv::Mat image, const std_msgs::msg::Header & header)
{
  size_t queue_size_after = 0;
  bool dropped = false;
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    const size_t limit = static_cast<size_t>(params_.queue_size);
    if (limit > 0 && image_queue_.size() >= limit) {
      if (!image_queue_.empty()) {
        image_queue_.pop_front();
        dropped = true;
        dropped_images_count_++;
      }
    }

    image_queue_.push_back({std::move(image), header});
    queue_size_after = image_queue_.size();
  }

  if (dropped) {
    RCLCPP_WARN(
      this->get_logger(),
      "Queue overflow: dropped oldest image (queue size: %zu, limit: %zu, total dropped: %zu)",
      queue_size_after,
      static_cast<size_t>(params_.queue_size),
      dropped_images_count_);
  }
}

void DeepObjectDetectionNode::onBatchTimer()
{
  if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    return;
  }

  std::vector<QueuedImage> batch;
  const size_t batch_size = static_cast<size_t>(params_.max_batch_size);

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (image_queue_.size() < batch_size) {
      return;
    }

    batch.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      batch.push_back(std::move(image_queue_.front()));
      image_queue_.pop_front();
    }
  }

  if (!batch.empty()) {
    try {
      processBatch(batch);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Exception in batch processing: %s", e.what());
    }
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

  auto start_time = std::chrono::steady_clock::now();
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
    RCLCPP_ERROR(this->get_logger(), "Packed input is empty after preprocessing %zu images", processed.size());
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

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  size_t dropped_count = 0;
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    dropped_count = dropped_images_count_;
  }

  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "Batch processing completed: %zu images in %" PRId64 " ms, total detections: %zu, dropped images: %zu",
    batch.size(),
    static_cast<int64_t>(elapsed_ms),
    std::accumulate(
      batch_detections.begin(),
      batch_detections.end(),
      size_t(0),
      [](size_t sum, const auto & dets) { return sum + dets.size(); }),
    dropped_count);

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
