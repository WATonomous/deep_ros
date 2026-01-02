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
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include "deep_object_detection/backend_manager.hpp"
#include "deep_object_detection/detection_msg_alias.hpp"
#include "deep_object_detection/generic_postprocessor.hpp"

#if __has_include(<deep_msgs/msg/multi_image.hpp>)
  #include <deep_msgs/msg/multi_image.hpp>
#endif

namespace deep_object_detection
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

DeepObjectDetectionNode::DeepObjectDetectionNode(const rclcpp::NodeOptions & options)
: LifecycleNode("deep_object_detection_node", options)
{
  declareAndReadParameters();
  RCLCPP_INFO(this->get_logger(), "Deep object detection node created, waiting for configuration");
}

void DeepObjectDetectionNode::declareAndReadParameters()
{
  // Model configuration
  params_.model_path = this->declare_parameter<std::string>("model_path", "");
  
  // Model metadata
  auto class_names_path = this->declare_parameter<std::string>("class_names_path", "");
  params_.model_metadata.num_classes = this->declare_parameter<int>("model.num_classes", 80);
  params_.model_metadata.class_names_file = class_names_path;
  auto bbox_format_str = this->declare_parameter<std::string>("model.bbox_format", "cxcywh");
  params_.model_metadata.bbox_format = stringToBboxFormat(bbox_format_str);
  
  // Preprocessing configuration
  params_.preprocessing.input_width = this->declare_parameter<int>("preprocessing.input_width", 640);
  params_.preprocessing.input_height = this->declare_parameter<int>("preprocessing.input_height", 640);
  auto normalization_type_str = this->declare_parameter<std::string>("preprocessing.normalization_type", "scale_0_1");
  params_.preprocessing.normalization_type = stringToNormalizationType(normalization_type_str);
  auto mean_d = this->declare_parameter<std::vector<double>>("preprocessing.mean", {0.0, 0.0, 0.0});
  auto std_d = this->declare_parameter<std::vector<double>>("preprocessing.std", {1.0, 1.0, 1.0});
  // Convert double vectors to float
  params_.preprocessing.mean.clear();
  params_.preprocessing.std.clear();
  for (auto v : mean_d) params_.preprocessing.mean.push_back(static_cast<float>(v));
  for (auto v : std_d) params_.preprocessing.std.push_back(static_cast<float>(v));
  auto resize_method_str = this->declare_parameter<std::string>("preprocessing.resize_method", "letterbox");
  params_.preprocessing.resize_method = stringToResizeMethod(resize_method_str);
  params_.preprocessing.keep_aspect_ratio = this->declare_parameter<bool>("preprocessing.keep_aspect_ratio", true);
  params_.preprocessing.pad_value = this->declare_parameter<int>("preprocessing.pad_value", 114);
  params_.preprocessing.color_format = this->declare_parameter<std::string>("preprocessing.color_format", "rgb");
  
  // Postprocessing configuration - always uses generic auto-detecting postprocessor
  params_.postprocessing.score_threshold = static_cast<float>(
    this->declare_parameter<double>("postprocessing.score_threshold", 0.25));
  params_.postprocessing.nms_iou_threshold = static_cast<float>(
    this->declare_parameter<double>("postprocessing.nms_iou_threshold", 0.45));
  params_.postprocessing.max_detections = this->declare_parameter<int>("postprocessing.max_detections", 300);
  auto score_activation_str = this->declare_parameter<std::string>("postprocessing.score_activation", "sigmoid");
  params_.postprocessing.score_activation = stringToScoreActivation(score_activation_str);
  
  // Multi-output model support
  params_.postprocessing.use_multi_output = this->declare_parameter<bool>("postprocessing.use_multi_output", false);
  params_.postprocessing.output_boxes_idx = this->declare_parameter<int>("postprocessing.output_boxes_idx", 0);
  params_.postprocessing.output_scores_idx = this->declare_parameter<int>("postprocessing.output_scores_idx", 1);
  params_.postprocessing.output_classes_idx = this->declare_parameter<int>("postprocessing.output_classes_idx", 2);
  
  // Topic configuration
  params_.input_image_topic = this->declare_parameter<std::string>("input_image_topic", "/camera/image_raw");
  camera_sync_topic_ = this->declare_parameter<std::string>("camera_sync_topic", "");
  // Check if use_camera_sync is explicitly set, otherwise infer from camera_sync_topic
  use_camera_sync_ = this->declare_parameter<bool>("use_camera_sync", false);
  if (!use_camera_sync_) {
    // If not explicitly set to true, infer from camera_sync_topic
    use_camera_sync_ = !camera_sync_topic_.empty();
  }
  // Declare camera_topics - handle empty arrays in YAML which may be parsed as null
  try {
    params_.camera_topics = this->declare_parameter<std::vector<std::string>>("camera_topics", std::vector<std::string>{});
  } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
    params_.camera_topics = this->get_parameter("camera_topics").as_string_array();
  } catch (const std::exception &) {
    // Empty array or null in YAML - use default empty vector
    params_.camera_topics = std::vector<std::string>{};
    // Declare the parameter manually if not already declared
    if (!this->has_parameter("camera_topics")) {
      this->declare_parameter("camera_topics", rclcpp::ParameterValue(std::vector<std::string>{}));
    }
  }
  params_.input_transport = this->declare_parameter<std::string>("input_transport", "raw");
  params_.input_qos_reliability = this->declare_parameter<std::string>("input_qos_reliability", "best_effort");
  params_.output_detections_topic = this->declare_parameter<std::string>("output_detections_topic", "/detections");
  
  // Batching configuration
  params_.batch_size_limit = this->declare_parameter<int>("batch_size_limit", 3);
  params_.queue_size = this->declare_parameter<int>("queue_size", 10);
  
  // Backend configuration
  params_.preferred_provider = this->declare_parameter<std::string>("preferred_provider", "tensorrt");
  params_.device_id = this->declare_parameter<int>("device_id", 0);
  params_.warmup_tensor_shapes = this->declare_parameter<bool>("warmup_tensor_shapes", true);
  params_.enable_trt_engine_cache = this->declare_parameter<bool>("enable_trt_engine_cache", false);
  params_.trt_engine_cache_path = this->declare_parameter<std::string>(
    "trt_engine_cache_path", "/tmp/deep_ros_ort_trt_cache");
  
  // Backward compatibility: support old parameter names
  // input_width/input_height without preprocessing prefix
  if (!this->has_parameter("preprocessing.input_width")) {
    auto old_width = this->declare_parameter<int>("input_width", 640);
    params_.preprocessing.input_width = old_width;
  }
  if (!this->has_parameter("preprocessing.input_height")) {
    auto old_height = this->declare_parameter<int>("input_height", 640);
    params_.preprocessing.input_height = old_height;
  }
  // use_letterbox -> preprocessing.resize_method
  if (this->declare_parameter<bool>("use_letterbox", false)) {
    params_.preprocessing.resize_method = ResizeMethod::LETTERBOX;
  }
  // score_threshold/nms_iou_threshold without prefix
  if (!this->has_parameter("postprocessing.score_threshold")) {
    params_.postprocessing.score_threshold = static_cast<float>(
      this->declare_parameter<double>("score_threshold", 0.25));
  }
  if (!this->has_parameter("postprocessing.nms_iou_threshold")) {
    params_.postprocessing.nms_iou_threshold = static_cast<float>(
      this->declare_parameter<double>("nms_iou_threshold", 0.45));
  }
  // Legacy parameter support (backward compatibility)
  auto preboxed_format = this->declare_parameter<std::string>("preboxed_format", "");
  if (!preboxed_format.empty()) {
    params_.model_metadata.bbox_format = stringToBboxFormat(preboxed_format);
  }
}

void DeepObjectDetectionNode::validateParameters()
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

  if (params_.preprocessing.input_width <= 0 || params_.preprocessing.input_height <= 0) {
    fail("preprocessing.input_width and preprocessing.input_height must be positive.");
  }
  if (params_.device_id < 0) {
    fail("device_id must be non-negative.");
  }
  if (params_.batch_size_limit < 1 || params_.batch_size_limit > 16) {
    fail("batch_size_limit must be between 1 and 16.");
  }
  if (params_.postprocessing.score_threshold <= 0.0 || params_.postprocessing.score_threshold > 1.0) {
    fail("postprocessing.score_threshold must be in (0.0, 1.0].");
  }
  if (params_.postprocessing.nms_iou_threshold <= 0.0 || params_.postprocessing.nms_iou_threshold > 1.0) {
    fail("postprocessing.nms_iou_threshold must be in (0.0, 1.0].");
  }
  if (params_.queue_size < 1) {
    fail("queue_size must be at least 1.");
  }

  if (use_camera_sync_) {
    // When using camera sync, we subscribe to MultiImage topic
    if (camera_sync_topic_.empty()) {
      fail("camera_sync_topic must be set when using camera sync node.");
    }
  } else {
    // When not using camera sync, we need individual camera topics
    if (params_.camera_topics.empty()) {
      fail("camera_topics must contain at least one compressed image topic (or set camera_sync_topic to use camera sync).");
    }
    bool all_compressed =
      std::all_of(params_.camera_topics.begin(), params_.camera_topics.end(), [this](const std::string & topic) {
        return isCompressedTopic(topic);
      });
    if (!all_compressed) {
      fail("camera_topics currently requires compressed image topics (suffix '/compressed' or '_compressed').");
    }
  }

  auto normalize_pref = params_.preferred_provider;
  std::transform(normalize_pref.begin(), normalize_pref.end(), normalize_pref.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (normalize_pref != "tensorrt" && normalize_pref != "cuda" && normalize_pref != "cpu") {
    fail("preferred_provider must be one of: tensorrt, cuda, cpu.");
  }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_configure(
  const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Configuring deep object detection node");

  try {
    validateParameters();

    loadClassNames();
    preprocessor_ = std::make_unique<ImagePreprocessor>(params_.preprocessing);
    backend_manager_ = std::make_unique<BackendManager>(*this, params_);

    backend_manager_->buildProviderOrder();
    if (!backend_manager_->initialize()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize any execution provider");
      return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
    }

    // Get output shape from model for auto-detection
    std::vector<size_t> output_shape;
    try {
      const size_t batch = static_cast<size_t>(params_.batch_size_limit);
      const size_t channels = 3;
      const size_t height = static_cast<size_t>(params_.preprocessing.input_height);
      const size_t width = static_cast<size_t>(params_.preprocessing.input_width);
      std::vector<size_t> input_shape = {batch, channels, height, width};
      output_shape = backend_manager_->getOutputShape(input_shape);
      if (!output_shape.empty()) {
        RCLCPP_INFO(
          this->get_logger(),
          "Detected model output shape: [%s]",
          formatShape(output_shape).c_str());
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        this->get_logger(),
        "Could not get output shape from model: %s. Will use auto-detection from first inference.",
        e.what());
    }

    // Create postprocessor with auto-detection
    const auto & config = params_.postprocessing;
    const auto & model_meta = params_.model_metadata;
    const bool use_letterbox = (params_.preprocessing.resize_method == ResizeMethod::LETTERBOX);
    
    GenericPostprocessor::OutputLayout layout;
    if (!output_shape.empty()) {
      layout = GenericPostprocessor::detectLayout(output_shape);
    } else {
      layout.auto_detect = true;
    }
    postprocessor_ = std::make_unique<GenericPostprocessor>(
      config, layout, model_meta.bbox_format, model_meta.num_classes,
      params_.class_names, use_letterbox);

    // Create publisher (will be activated in on_activate)
    detection_pub_ =
      this->create_publisher<Detection2DArrayMsg>(params_.output_detections_topic, rclcpp::SensorDataQoS{});

    if (use_camera_sync_) {
      multi_camera_mode_ = true;
      RCLCPP_INFO(this->get_logger(), "Using camera sync mode - will subscribe to: %s", camera_sync_topic_.c_str());
    } else {
      if (params_.camera_topics.empty()) {
        RCLCPP_ERROR(this->get_logger(), "camera_topics must contain at least one topic for multi-camera mode");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
      }
      multi_camera_mode_ = true;
    }

    std::string shape_str = output_shape.empty() ? "auto-detect" : formatShape(output_shape);
    RCLCPP_INFO(
      this->get_logger(),
      "Deep object detection node configured. Model: %s, input size: %dx%d, batch limit: %d, provider: %s, postprocessor: %s, output shape: [%s]",
      params_.model_path.c_str(),
      params_.preprocessing.input_width,
      params_.preprocessing.input_height,
      params_.batch_size_limit,
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
  const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Activating deep object detection node");

  try {
    // Publishers are automatically activated for lifecycle nodes
    // Setup subscriptions
    if (use_camera_sync_) {
      setupCameraSyncSubscription();
    } else {
      setupMultiCameraSubscriptions();
    }

    // Start batch processing timer
    batch_timer_ = this->create_wall_timer(chrono::milliseconds(5), std::bind(&DeepObjectDetectionNode::onBatchTimer, this));

    RCLCPP_INFO(this->get_logger(), "Deep object detection node activated and ready to process images");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to activate: %s", e.what());
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
  }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_deactivate(
  const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Deactivating deep object detection node");

  // Stop timer
  if (batch_timer_) {
    batch_timer_->cancel();
    batch_timer_.reset();
  }

  // Clear subscriptions
  multi_camera_subscriptions_.clear();
#if __has_include(<deep_msgs/msg/multi_image.hpp>)
  if (multi_image_sub_) {
    multi_image_sub_.reset();
  }
#endif
  // Only shutdown image_sub_ if not using camera sync (it may not have been initialized)
  if (!use_camera_sync_) {
    image_sub_.shutdown();
  }
  if (compressed_image_sub_) {
    compressed_image_sub_.reset();
  }

  // Clear image queue
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    image_queue_.clear();
  }

  // Publishers are automatically deactivated for lifecycle nodes

  RCLCPP_INFO(this->get_logger(), "Deep object detection node deactivated");
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_cleanup(
  const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Cleaning up deep object detection node");

  // Release backend resources
  backend_manager_.reset();

  // Release preprocessor and postprocessor
  preprocessor_.reset();
  postprocessor_.reset();

  // Clear class names
  params_.class_names.clear();

  // Reset publisher
  detection_pub_.reset();

  RCLCPP_INFO(this->get_logger(), "Deep object detection node cleaned up");
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn DeepObjectDetectionNode::on_shutdown(
  const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Shutting down deep object detection node");

  // Stop timer if still active
  if (batch_timer_) {
    batch_timer_->cancel();
    batch_timer_.reset();
  }

  // Clear subscriptions
  multi_camera_subscriptions_.clear();
#if __has_include(<deep_msgs/msg/multi_image.hpp>)
  if (multi_image_sub_) {
    multi_image_sub_.reset();
  }
#endif
  // Only shutdown image_sub_ if not using camera sync (it may not have been initialized)
  if (!use_camera_sync_) {
    image_sub_.shutdown();
  }
  if (compressed_image_sub_) {
    compressed_image_sub_.reset();
  }

  // Clear image queue
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    image_queue_.clear();
  }

  // Release all resources
  backend_manager_.reset();
  preprocessor_.reset();
  postprocessor_.reset();
  params_.class_names.clear();
  detection_pub_.reset();

  RCLCPP_INFO(this->get_logger(), "Deep object detection node shut down");
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

void DeepObjectDetectionNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    auto converted = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    enqueueImage(std::move(converted->image), msg->header);
  } catch (const std::exception & e) {
    RCLCPP_WARN(this->get_logger(), "Failed to convert raw image: %s", e.what());
  }
}

void DeepObjectDetectionNode::onCompressedImage(const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg)
{
  handleCompressedImage(msg, -1);
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
        this->handleCompressedImage(msg, static_cast<int>(i));
      });
    multi_camera_subscriptions_.push_back(sub);
    RCLCPP_INFO(this->get_logger(), "Subscribed to compressed image topic %s (camera %zu)", topic.c_str(), i);
  }
}

void DeepObjectDetectionNode::setupCameraSyncSubscription()
{
#if __has_include(<deep_msgs/msg/multi_image.hpp>)
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
#else
  RCLCPP_ERROR(this->get_logger(), "deep_msgs not available - cannot subscribe to MultiImage topic");
  throw std::runtime_error("deep_msgs package required for camera sync integration");
#endif
}

void DeepObjectDetectionNode::onMultiImage(const deep_msgs::msg::MultiImage::ConstSharedPtr & msg)
{
#if __has_include(<deep_msgs/msg/multi_image.hpp>)
  try {
    // Extract all images from the MultiImage message and enqueue them
    // The MultiImage message contains an array of synchronized CompressedImage messages
    for (size_t i = 0; i < msg->images.size(); ++i) {
      const auto & compressed_img = msg->images[i];
      handleCompressedImage(
        std::make_shared<sensor_msgs::msg::CompressedImage>(compressed_img), static_cast<int>(i));
    }
    RCLCPP_DEBUG(
      this->get_logger(),
      "Processed MultiImage message with %zu synchronized images",
      msg->images.size());
  } catch (const std::exception & e) {
    RCLCPP_WARN(this->get_logger(), "Failed to process MultiImage message: %s", e.what());
  }
#else
  (void)msg;
  RCLCPP_ERROR(this->get_logger(), "deep_msgs not available - cannot process MultiImage");
#endif
}

void DeepObjectDetectionNode::handleCompressedImage(
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

void DeepObjectDetectionNode::enqueueImage(cv::Mat image, const std_msgs::msg::Header & header)
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

size_t DeepObjectDetectionNode::queueLimit() const
{
  return static_cast<size_t>(params_.queue_size);
}

bool DeepObjectDetectionNode::isCompressedTopic(const std::string & topic) const
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

void DeepObjectDetectionNode::processBatch(const std::vector<QueuedImage> & batch)
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

    std::vector<std::vector<SimpleDetection>> batch_detections;
    if (params_.postprocessing.use_multi_output) {
      // Multi-output model: get all outputs and decode separately
      auto output_tensors = backend_manager_->inferAllOutputs(packed_input);
      batch_detections = postprocessor_->decodeMultiOutput(output_tensors, metas);
    } else {
      // Single output model: use standard decode
      auto output_tensor = backend_manager_->infer(packed_input);
      batch_detections = postprocessor_->decode(output_tensor, metas);
    }
    for (size_t i = 0; i < batch_detections.size(); ++i) {
      RCLCPP_INFO(this->get_logger(), "Image %zu: %zu detections", i, batch_detections[i].size());
    }
    publishDetections(batch_detections, headers, metas);
    RCLCPP_INFO(this->get_logger(), "Published detections for %zu images", batch_detections.size());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Batch processing failed: %s", e.what());
  }
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
    RCLCPP_WARN(this->get_logger(), "Failed to open class_names_path: %s", class_names_path.c_str());
    return;
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

