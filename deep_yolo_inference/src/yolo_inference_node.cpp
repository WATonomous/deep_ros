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

#include "deep_yolo_inference/yolo_inference_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <image_transport/image_transport.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include "deep_yolo_inference/backend_manager.hpp"
#include "deep_yolo_inference/detection_msg_alias.hpp"
#include "deep_yolo_inference/processing.hpp"

namespace deep_yolo_inference
{

namespace chrono = std::chrono;

class ScopeExit
{
public:
  explicit ScopeExit(std::function<void()> fn)
  : fn_(std::move(fn))
  {}

  ~ScopeExit()
  {
    if (fn_) {
      fn_();
    }
  }

private:
  std::function<void()> fn_;
};

YoloInferenceNode::YoloInferenceNode(const rclcpp::NodeOptions & options)
: Node("yolo_inference_node", options)
{
  declareAndReadParameters();
  validateParameters();

  loadClassNames();
  preprocessor_ = std::make_unique<ImagePreprocessor>(params_);
  postprocessor_ = std::make_unique<Postprocessor>(params_, class_names_);
  backend_manager_ = std::make_unique<BackendManager>(*this, params_);

  backend_manager_->buildProviderOrder();
  if (!backend_manager_->initialize()) {
    throw std::runtime_error("Failed to initialize any execution provider");
  }
  detection_pub_ =
    this->create_publisher<Detection2DArrayMsg>(params_.output_detections_topic, rclcpp::SensorDataQoS{});

  if (params_.camera_topics.empty()) {
    throw std::runtime_error("camera_topics must contain at least one topic for multi-camera mode");
  }
  multi_camera_mode_ = true;
  setupMultiCameraSubscriptions();

  batch_timer_ = this->create_wall_timer(chrono::milliseconds(5), std::bind(&YoloInferenceNode::onBatchTimer, this));

  RCLCPP_INFO(
    this->get_logger(),
    "YOLO inference node initialized. Model: %s, input size: %dx%d, batch limit: %d, provider: %s",
    params_.model_path.c_str(),
    params_.input_width,
    params_.input_height,
    params_.batch_size_limit,
    backend_manager_->activeProvider().c_str());
}

void YoloInferenceNode::declareAndReadParameters()
{
  params_.model_path = this->declare_parameter<std::string>("model_path", "");
  params_.input_image_topic = this->declare_parameter<std::string>("input_image_topic", params_.input_image_topic);
  params_.camera_topics = this->declare_parameter<std::vector<std::string>>("camera_topics", params_.camera_topics);
  params_.input_transport = this->declare_parameter<std::string>("input_transport", params_.input_transport);
  params_.input_qos_reliability =
    this->declare_parameter<std::string>("input_qos_reliability", params_.input_qos_reliability);
  params_.output_detections_topic =
    this->declare_parameter<std::string>("output_detections_topic", params_.output_detections_topic);
  params_.input_width = this->declare_parameter<int>("input_width", params_.input_width);
  params_.input_height = this->declare_parameter<int>("input_height", params_.input_height);
  params_.use_letterbox = this->declare_parameter<bool>("use_letterbox", params_.use_letterbox);
  params_.batch_size_limit = this->declare_parameter<int>("batch_size_limit", params_.batch_size_limit);
  params_.score_threshold = this->declare_parameter<double>("score_threshold", params_.score_threshold);
  params_.nms_iou_threshold = this->declare_parameter<double>("nms_iou_threshold", params_.nms_iou_threshold);
  params_.preferred_provider = this->declare_parameter<std::string>("preferred_provider", params_.preferred_provider);
  params_.device_id = this->declare_parameter<int>("device_id", params_.device_id);
  params_.class_names_path = this->declare_parameter<std::string>("class_names_path", params_.class_names_path);
  params_.queue_size = this->declare_parameter<int>("queue_size", params_.queue_size);
  params_.warmup_tensor_shapes = this->declare_parameter<bool>("warmup_tensor_shapes", params_.warmup_tensor_shapes);
  params_.enable_trt_engine_cache =
    this->declare_parameter<bool>("enable_trt_engine_cache", params_.enable_trt_engine_cache);
  params_.trt_engine_cache_path =
    this->declare_parameter<std::string>("trt_engine_cache_path", params_.trt_engine_cache_path);
  params_.preboxed_format = this->declare_parameter<std::string>("preboxed_format", params_.preboxed_format);
}

void YoloInferenceNode::validateParameters()
{
  auto fail = [this](const std::string & msg) {
    RCLCPP_ERROR(this->get_logger(), "%s", msg.c_str());
    throw std::runtime_error(msg);
  };

  auto normalize_transport = params_.input_transport;
  std::transform(
    normalize_transport.begin(), normalize_transport.end(), normalize_transport.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });

  if (normalize_transport != "raw" && normalize_transport != "compressed") {
    fail("input_transport must be either 'raw' or 'compressed'.");
  }
  params_.input_transport = normalize_transport;

  auto reliability = params_.input_qos_reliability;
  std::transform(reliability.begin(), reliability.end(), reliability.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (reliability != "best_effort" && reliability != "reliable") {
    fail("input_qos_reliability must be either 'best_effort' or 'reliable'.");
  }
  params_.input_qos_reliability = reliability;

  if (params_.model_path.empty()) {
    fail("Parameter 'model_path' must be set to a valid ONNX model file.");
  } else {
    std::filesystem::path model_path(params_.model_path);
    if (!std::filesystem::exists(model_path)) {
      fail("Model path does not exist: " + params_.model_path);
    }
  }

  if (params_.input_width <= 0 || params_.input_height <= 0) {
    fail("input_width and input_height must be positive.");
  }
  if (params_.device_id < 0) {
    fail("device_id must be non-negative.");
  }
  if (params_.batch_size_limit < 1 || params_.batch_size_limit > 6) {
    fail("batch_size_limit must be between 1 and 6.");
  }
  if (params_.score_threshold <= 0.0 || params_.score_threshold > 1.0) {
    fail("score_threshold must be in (0.0, 1.0].");
  }
  if (params_.nms_iou_threshold <= 0.0 || params_.nms_iou_threshold > 1.0) {
    fail("nms_iou_threshold must be in (0.0, 1.0].");
  }
  if (params_.queue_size < 1) {
    fail("queue_size must be at least 1.");
  }

  if (params_.camera_topics.empty()) {
    fail("camera_topics must contain at least one compressed image topic.");
  }
  bool all_compressed =
    std::all_of(params_.camera_topics.begin(), params_.camera_topics.end(), [this](const std::string & topic) {
      return isCompressedTopic(topic);
    });
  if (!all_compressed) {
    fail("camera_topics currently requires compressed image topics (suffix '/compressed' or '_compressed').");
  }

  auto normalize_pref = params_.preferred_provider;
  std::transform(normalize_pref.begin(), normalize_pref.end(), normalize_pref.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (normalize_pref != "tensorrt" && normalize_pref != "cuda" && normalize_pref != "cpu") {
    fail("preferred_provider must be one of: tensorrt, cuda, cpu.");
  }

  auto normalize_preboxed = params_.preboxed_format;
  std::transform(normalize_preboxed.begin(), normalize_preboxed.end(), normalize_preboxed.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (normalize_preboxed != "cxcywh" && normalize_preboxed != "xyxy") {
    fail("preboxed_format must be either 'cxcywh' or 'xyxy'.");
  }
  params_.preboxed_format = normalize_preboxed;
}

void YoloInferenceNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    auto converted = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    enqueueImage(std::move(converted->image), msg->header);
  } catch (const std::exception & e) {
    RCLCPP_WARN(this->get_logger(), "Failed to convert raw image: %s", e.what());
  }
}

void YoloInferenceNode::onCompressedImage(const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg)
{
  handleCompressedImage(msg, -1);
}

void YoloInferenceNode::setupMultiCameraSubscriptions()
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
        this->handleCompressedImage(msg, static_cast<int>(i));
      });
    multi_camera_subscriptions_.push_back(sub);
    RCLCPP_INFO(this->get_logger(), "Subscribed to compressed image topic %s (camera %zu)", topic.c_str(), i);
  }
}

void YoloInferenceNode::handleCompressedImage(
  const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg, int camera_id)
{
  try {
    cv::Mat compressed_mat(
      1, static_cast<int>(msg->data.size()), CV_8UC1, const_cast<unsigned char *>(msg->data.data()));
    cv::Mat decoded = cv::imdecode(compressed_mat, cv::IMREAD_COLOR);
    if (decoded.empty()) {
      RCLCPP_WARN(this->get_logger(), "Failed to decode compressed image; skipping frame");
      return;
    }

    enqueueImage(std::move(decoded), msg->header);
    if (camera_id >= 0) {
      RCLCPP_DEBUG(
        this->get_logger(), "Enqueued frame from camera %d (topic %s)", camera_id, msg->header.frame_id.c_str());
    }
  } catch (const std::exception & e) {
    RCLCPP_WARN(this->get_logger(), "Skipping compressed image due to error: %s", e.what());
  }
}

void YoloInferenceNode::enqueueImage(cv::Mat image, const std_msgs::msg::Header & header)
{
  std::lock_guard<std::mutex> lock(queue_mutex_);
  image_queue_.push_back({std::move(image), header});
  const auto limit = queueLimit();
  if (limit > 0 && image_queue_.size() > limit) {
    image_queue_.pop_front();
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000, "Image queue exceeded limit (%zu); dropping oldest frame", limit);
  }
  RCLCPP_DEBUG(
    this->get_logger(),
    "Image received. Queue size: %zu, Stamp: %.6f",
    image_queue_.size(),
    rclcpp::Time(header.stamp).seconds());
  RCLCPP_INFO_ONCE(this->get_logger(), "Received first image from configured camera topics");
}

size_t YoloInferenceNode::queueLimit() const
{
  return static_cast<size_t>(params_.queue_size);
}

bool YoloInferenceNode::isCompressedTopic(const std::string & topic) const
{
  static const std::string slash_suffix{"/compressed"};
  static const std::string underscore_suffix{"_compressed"};

  if (
    topic.size() >= slash_suffix.size() &&
    topic.compare(topic.size() - slash_suffix.size(), slash_suffix.size(), slash_suffix) == 0)
  {
    return true;
  }

  if (
    topic.size() >= underscore_suffix.size() &&
    topic.compare(topic.size() - underscore_suffix.size(), underscore_suffix.size(), underscore_suffix) == 0)
  {
    return true;
  }

  return false;
}

std::string YoloInferenceNode::formatShape(const std::vector<size_t> & shape) const
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

void YoloInferenceNode::onBatchTimer()
{
  if (processing_.exchange(true)) {
    return;
  }
  ScopeExit guard([this]() { processing_.store(false); });

  std::vector<QueuedImage> batch;
  const size_t required = static_cast<size_t>(params_.batch_size_limit);
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (image_queue_.size() < required) {
      return;
    }

    batch.reserve(required);
    for (size_t i = 0; i < required; ++i) {
      batch.push_back(image_queue_.front());
      image_queue_.pop_front();
    }
  }

  if (!batch.empty()) {
    processBatch(batch);
  }
}

void YoloInferenceNode::processBatch(const std::vector<QueuedImage> & batch)
{
  if (!backend_manager_ || !backend_manager_->hasExecutor()) {
    RCLCPP_WARN(this->get_logger(), "No backend initialized; dropping batch");
    return;
  }
  try {
    RCLCPP_INFO(this->get_logger(), "Processing batch of size: %zu", batch.size());
    std::vector<cv::Mat> processed;
    std::vector<ImageMeta> metas;
    std::vector<std_msgs::msg::Header> headers;
    processed.reserve(batch.size());
    metas.reserve(batch.size());
    headers.reserve(batch.size());

    for (const auto & item : batch) {
      try {
        ImageMeta meta;
        cv::Mat preprocessed = preprocessor_->preprocess(item.bgr, meta);
        processed.push_back(std::move(preprocessed));
        metas.push_back(meta);
        headers.push_back(item.header);
        RCLCPP_DEBUG(
          this->get_logger(),
          "Preprocessed image size: %dx%d (orig %dx%d)",
          processed.back().cols,
          processed.back().rows,
          meta.original_width,
          meta.original_height);
      } catch (const std::exception & e) {
        RCLCPP_WARN(this->get_logger(), "Skipping image due to preprocessing error: %s", e.what());
      }
    }

    if (processed.empty()) {
      RCLCPP_WARN(this->get_logger(), "No valid images to process in this batch");
      return;
    }

    const auto & packed_input = preprocessor_->pack(processed);
    if (packed_input.data.empty()) {
      RCLCPP_WARN(this->get_logger(), "Packed input is empty; skipping inference");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Input tensor shape about to build: [%s]", formatShape(packed_input.shape).c_str());

    auto output_tensor = backend_manager_->infer(packed_input);
    auto batch_detections = postprocessor_->decode(output_tensor, metas);
    for (size_t i = 0; i < batch_detections.size(); ++i) {
      RCLCPP_INFO(this->get_logger(), "Image %zu: %zu detections", i, batch_detections[i].size());
    }
    publishDetections(batch_detections, headers, metas);
    RCLCPP_INFO(this->get_logger(), "Published detections for %zu images", batch_detections.size());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Batch processing failed: %s", e.what());
  }
}

void YoloInferenceNode::publishDetections(
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

void YoloInferenceNode::loadClassNames()
{
  if (params_.class_names_path.empty()) {
    return;
  }

  class_names_.clear();

  std::ifstream file(params_.class_names_path);
  if (!file.is_open()) {
    RCLCPP_WARN(this->get_logger(), "Failed to open class_names_path: %s", params_.class_names_path.c_str());
    return;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty()) {
      class_names_.push_back(line);
    }
  }

  RCLCPP_INFO(
    this->get_logger(), "Loaded %zu class names from %s", class_names_.size(), params_.class_names_path.c_str());
}

std::shared_ptr<rclcpp::Node> createYoloInferenceNode(const rclcpp::NodeOptions & options)
{
  return std::make_shared<YoloInferenceNode>(options);
}

}  // namespace deep_yolo_inference

RCLCPP_COMPONENTS_REGISTER_NODE(deep_yolo_inference::YoloInferenceNode)
