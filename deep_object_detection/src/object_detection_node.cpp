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

#include "deep_object_detection/object_detection_node.hpp"

#include <cv_bridge/cv_bridge.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/string.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include "deep_msgs/msg/multi_image.hpp"
#include "deep_msgs/msg/multi_image_raw.hpp"
#include "deep_object_detection/inference_interface.hpp"

namespace deep_object_detection
{

ObjectDetectionNode::ObjectDetectionNode(const rclcpp::NodeOptions & options)
: Node("object_detection_node", options)
, total_inferences_(0)
, average_inference_time_(0.0)
{
  initializeParameters();
  initializeInference();
  initializePublishers();

  RCLCPP_INFO(this->get_logger(), "Object Detection Node initialized");
}

ObjectDetectionNode::~ObjectDetectionNode() = default;

void ObjectDetectionNode::initializeParameters()
{
  // Model parameters
  config_.model_path = this->declare_parameter<std::string>("model_path", "./model.onnx");
  config_.max_batch_size = this->declare_parameter<int>("max_batch_size", 1);
  config_.inference_rate = this->declare_parameter<double>("inference_rate", 30.0);

  config_.model_path = this->get_parameter("model_path").as_string();
  RCLCPP_INFO(this->get_logger(), "Model path: %s", config_.model_path.c_str());
  config_.max_batch_size = this->get_parameter("max_batch_size").as_int();
  config_.inference_rate = this->get_parameter("inference_rate").as_double();

  // Log the actual batch size being used
  RCLCPP_INFO(this->get_logger(), "Max batch size: %d", config_.max_batch_size);

  // Camera parameters
  config_.image_topic = this->declare_parameter<std::string>("image_topic", "/front/image_compressed");
  std::string topic_type_str = this->declare_parameter<std::string>("topic_type", "compressed_image");

  config_.image_topic = this->get_parameter("image_topic").as_string();
  topic_type_str = this->get_parameter("topic_type").as_string();

  // Parse topic type
  if (topic_type_str == "raw_image") {
    config_.topic_type = ImageTopicType::RAW_IMAGE;
  } else if (topic_type_str == "compressed_image") {
    config_.topic_type = ImageTopicType::COMPRESSED_IMAGE;
  } else if (topic_type_str == "multi_image") {
    config_.topic_type = ImageTopicType::MULTI_IMAGE;
  } else if (topic_type_str == "multi_image_raw") {
    config_.topic_type = ImageTopicType::MULTI_IMAGE_RAW;
  } else {
    RCLCPP_WARN(this->get_logger(), "Unknown topic type '%s', defaulting to compressed_image", topic_type_str.c_str());
    config_.topic_type = ImageTopicType::COMPRESSED_IMAGE;
  }

  config_.queue_size = this->declare_parameter<int>("queue_size", 10);

  config_.queue_size = this->get_parameter("queue_size").as_int();

  // Output parameters
  config_.detection_topic = this->declare_parameter<std::string>("detection_topic", "/detections");
  config_.visualization_topic = this->declare_parameter<std::string>("visualization_topic", "/detection_markers");
  config_.enable_visualization = this->declare_parameter<bool>("enable_visualization", true);
  config_.enable_debug = this->declare_parameter<bool>("enable_debug", false);

  config_.detection_topic = this->get_parameter("detection_topic").as_string();
  config_.visualization_topic = this->get_parameter("visualization_topic").as_string();
  config_.enable_visualization = this->get_parameter("enable_visualization").as_bool();
  config_.enable_debug = this->get_parameter("enable_debug").as_bool();

  // Inference configuration
  config_.inference_config.model_path = config_.model_path;
  config_.inference_config.input_width = this->declare_parameter<int>("input_width", 640);
  config_.inference_config.input_height = this->declare_parameter<int>("input_height", 640);
  config_.inference_config.confidence_threshold = this->declare_parameter<double>("confidence_threshold", 0.5);
  config_.inference_config.nms_threshold = this->declare_parameter<double>("nms_threshold", 0.4);
  config_.inference_config.max_batch_size = config_.max_batch_size;
  config_.inference_config.use_gpu = this->declare_parameter<bool>("use_gpu", true);
  config_.inference_config.input_blob_name = this->declare_parameter<std::string>("input_blob_name", "images");
  config_.inference_config.output_blob_name = this->declare_parameter<std::string>("output_blob_name", "output0");

  // Configure inference backend
  std::string backend_str = this->declare_parameter<std::string>("inference_backend", "ort_backend");
  if (backend_str == "ort_backend") {
    config_.inference_config.backend = InferenceBackend::ORT_BACKEND;
  } else {
    config_.inference_config.backend = InferenceBackend::AUTO;
  }

  config_.inference_config.input_width = this->get_parameter("input_width").as_int();
  config_.inference_config.input_height = this->get_parameter("input_height").as_int();
  config_.inference_config.confidence_threshold = this->get_parameter("confidence_threshold").as_double();
  config_.inference_config.nms_threshold = this->get_parameter("nms_threshold").as_double();
  config_.inference_config.use_gpu = this->get_parameter("use_gpu").as_bool();
  config_.inference_config.input_blob_name = this->get_parameter("input_blob_name").as_string();
  config_.inference_config.output_blob_name = this->get_parameter("output_blob_name").as_string();

  // Load class names
  auto class_names_param = this->declare_parameter<std::vector<std::string>>("class_names", std::vector<std::string>{});

  config_.inference_config.class_names = this->get_parameter("class_names").as_string_array();

  RCLCPP_INFO(
    this->get_logger(),
    "Initialized with topic '%s' (type: %s) and %zu classes",
    config_.image_topic.c_str(),
    topic_type_str.c_str(),
    config_.inference_config.class_names.size());
}

void ObjectDetectionNode::initializeInference()
{
  inference_engine_ = createInferenceEngine(config_.inference_config);

  if (!inference_engine_->initialize()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize inference engine");
    throw std::runtime_error("Inference engine initialization failed");
  }

  RCLCPP_INFO(this->get_logger(), "Inference engine initialized successfully");
}

void ObjectDetectionNode::initializeSubscribers()
{
  auto qos = rclcpp::QoS(rclcpp::KeepLast(config_.queue_size)).best_effort();
  switch (config_.topic_type) {
    case ImageTopicType::RAW_IMAGE:
      image_transport_ = std::make_unique<image_transport::ImageTransport>(shared_from_this());
      image_transport_subscriber_ = std::make_shared<image_transport::Subscriber>(image_transport_->subscribe(
        config_.image_topic,
        config_.queue_size,
        std::bind(&ObjectDetectionNode::rawImageCallback, this, std::placeholders::_1)));
      RCLCPP_INFO(this->get_logger(), "Subscribed to raw image topic: %s", config_.image_topic.c_str());
      break;

    case ImageTopicType::COMPRESSED_IMAGE:
      image_subscriber_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        config_.image_topic,
        qos,
        std::bind(&ObjectDetectionNode::compressedImageCallback, this, std::placeholders::_1));
      RCLCPP_INFO(this->get_logger(), "Subscribed to compressed image topic: %s", config_.image_topic.c_str());
      break;

    case ImageTopicType::MULTI_IMAGE:
      image_subscriber_ = this->create_subscription<deep_msgs::msg::MultiImage>(
        config_.image_topic,
        config_.queue_size,
        std::bind(&ObjectDetectionNode::multiImageCallback, this, std::placeholders::_1));
      RCLCPP_INFO(this->get_logger(), "Subscribed to multi image topic: %s", config_.image_topic.c_str());
      break;

    case ImageTopicType::MULTI_IMAGE_RAW:
      image_subscriber_ = this->create_subscription<deep_msgs::msg::MultiImageRaw>(
        config_.image_topic,
        config_.queue_size,
        std::bind(&ObjectDetectionNode::multiImageRawCallback, this, std::placeholders::_1));
      RCLCPP_INFO(this->get_logger(), "Subscribed to multi image raw topic: %s", config_.image_topic.c_str());
      break;
  }
}

void ObjectDetectionNode::initializePublishers()
{
  detection_publisher_ =
    this->create_publisher<vision_msgs::msg::Detection2DArray>(config_.detection_topic, config_.queue_size);

  if (config_.enable_visualization) {
    visualization_publisher_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(config_.visualization_topic, config_.queue_size);

    // New: image topic for visualization (annotated images)
    std::string image_vis_topic = config_.visualization_topic + std::string("/image");
    visualization_image_publisher_ =
      this->create_publisher<sensor_msgs::msg::Image>(image_vis_topic, config_.queue_size);
  }

  if (config_.enable_debug) {
    performance_publisher_ = this->create_publisher<std_msgs::msg::String>("/performance_stats", 10);
  }
}

void ObjectDetectionNode::setupTimers()
{
  auto timer_period = std::chrono::duration<double>(1.0 / config_.inference_rate);
  batch_timer_ = this->create_wall_timer(timer_period, std::bind(&ObjectDetectionNode::batchInferenceCallback, this));
}

void ObjectDetectionNode::rawImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    addImageToBatch(cv_ptr->image, msg->header);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
}

void ObjectDetectionNode::compressedImageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg)
{
  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    if (cv_ptr && !cv_ptr->image.empty()) {
      addImageToBatch(cv_ptr->image, msg->header);
    } else {
      RCLCPP_WARN(this->get_logger(), "Received empty compressed image");
    }
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception for compressed image: %s", e.what());
  }
}

void ObjectDetectionNode::multiImageCallback(const deep_msgs::msg::MultiImage::ConstSharedPtr & msg)
{
  try {
    std::vector<cv::Mat> images;
    images.reserve(msg->images.size());

    for (const auto & compressed_img : msg->images) {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(compressed_img, sensor_msgs::image_encodings::BGR8);
      images.emplace_back(std::move(cv_ptr->image));
    }

    addImagesToBatch(images, msg->header);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception for multiImage compressed: %s", e.what());
  }
}

void ObjectDetectionNode::multiImageRawCallback(const deep_msgs::msg::MultiImageRaw::ConstSharedPtr & msg)
{
  try {
    std::vector<cv::Mat> images;
    images.reserve(msg->images.size());

    for (const auto & raw_img : msg->images) {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(raw_img, sensor_msgs::image_encodings::BGR8);
      images.push_back(cv_ptr->image);
    }

    addImagesToBatch(images, msg->header);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception for multiImage raw: %s", e.what());
  }
}

void ObjectDetectionNode::addImageToBatch(const cv::Mat & image, const std_msgs::msg::Header & header, int camera_id)
{
  std::lock_guard<std::mutex> lock(batch_mutex_);

  current_batch_.images.push_back(image);
  current_batch_.headers.push_back(header);
  current_batch_.camera_ids.push_back(camera_id);
  current_batch_.timestamp = std::chrono::steady_clock::now();

  if (config_.enable_debug) {
    RCLCPP_DEBUG(
      this->get_logger(),
      "Added image to batch. Current batch size: %zu, Max batch size: %d",
      current_batch_.images.size(),
      config_.max_batch_size);
  }

  // If max_batch_size is 1, process immediately (don't wait for timer)
  if (config_.max_batch_size == 1 && current_batch_.size() >= 1) {
    if (config_.enable_debug) {
      RCLCPP_DEBUG(this->get_logger(), "Processing single image immediately");
    }

    ImageBatch batch_to_process = std::move(current_batch_);
    current_batch_.clear();

    // Release lock before processing to avoid blocking other callbacks
    lock.~lock_guard();
    processBatch(batch_to_process);
  }
}

void ObjectDetectionNode::addImagesToBatch(const std::vector<cv::Mat> & images, const std_msgs::msg::Header & header)
{
  // If max_batch_size is 1, process each image individually
  if (config_.max_batch_size == 1) {
    for (size_t i = 0; i < images.size(); ++i) {
      addImageToBatch(images[i], header, static_cast<int>(i));
      // Each call to addImageToBatch will process immediately due to batch size 1
    }
    return;
  }

  // Normal batching logic for batch size > 1
  std::lock_guard<std::mutex> lock(batch_mutex_);

  for (size_t i = 0; i < images.size(); ++i) {
    current_batch_.images.push_back(images[i]);
    current_batch_.headers.push_back(header);
    current_batch_.camera_ids.push_back(static_cast<int>(i));  // Use index as camera ID
  }
  current_batch_.timestamp = std::chrono::steady_clock::now();
}

bool ObjectDetectionNode::shouldProcessBatch() const
{
  if (current_batch_.empty()) {
    return false;
  }

  // Process if batch is full or if timeout reached
  bool batch_full = current_batch_.size() >= static_cast<size_t>(config_.max_batch_size);

  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - current_batch_.timestamp);
  bool timeout = elapsed.count() > 50;  // 50ms timeout

  if (config_.enable_debug && (batch_full || timeout)) {
    RCLCPP_DEBUG(
      this->get_logger(),
      "Processing batch: size=%zu, max=%d, batch_full=%s, timeout=%s",
      current_batch_.size(),
      config_.max_batch_size,
      batch_full ? "true" : "false",
      timeout ? "true" : "false");
  }

  return batch_full || timeout;
}

void ObjectDetectionNode::batchInferenceCallback()
{
  // Skip timer-based processing if max_batch_size is 1 (immediate processing)
  if (config_.max_batch_size == 1) {
    return;
  }

  std::unique_lock<std::mutex> lock(batch_mutex_);

  if (!shouldProcessBatch()) {
    return;
  }

  ImageBatch batch_to_process = std::move(current_batch_);
  current_batch_.clear();
  lock.unlock();

  processBatch(batch_to_process);
}

void ObjectDetectionNode::processBatch(const ImageBatch & batch_to_process)
{
  if (batch_to_process.empty()) {
    return;
  }

  auto start_time = std::chrono::steady_clock::now();

  if (config_.enable_debug) {
    RCLCPP_INFO(
      this->get_logger(),
      "Processing batch with %zu images, max_batch_size: %d",
      batch_to_process.images.size(),
      config_.max_batch_size);
  }

  // Run inference
  if (inference_engine_ == nullptr) {
    RCLCPP_ERROR(this->get_logger(), "Inference engine is not initialized");
    return;
  }
  auto batch_detections = inference_engine_->inferBatch(batch_to_process.images);

  auto end_time = std::chrono::steady_clock::now();
  auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  if (config_.enable_debug) {
    RCLCPP_INFO(
      this->get_logger(),
      "Inference completed in %ld ms for batch of %zu images",
      inference_time.count(),
      batch_to_process.images.size());
  }

  // Update performance statistics
  total_inferences_++;
  double current_time = inference_time.count();
  average_inference_time_ = (average_inference_time_ * (total_inferences_ - 1) + current_time) / total_inferences_;

  if (config_.enable_debug) {
    RCLCPP_DEBUG(
      this->get_logger(),
      "Batch inference time: %ld ms, batch size: %zu",
      inference_time.count(),
      batch_to_process.size());
    publishPerformanceStats();
  }

  // Publish results
  publishDetections(batch_detections, batch_to_process.headers);

  if (config_.enable_visualization) {
    publishVisualization(
      batch_detections, batch_to_process.images, batch_to_process.headers, batch_to_process.camera_ids);
  }
}

void ObjectDetectionNode::publishDetections(
  const std::vector<std::vector<Detection>> & batch_detections, const std::vector<std_msgs::msg::Header> & headers)
{
  for (size_t i = 0; i < batch_detections.size() && i < headers.size(); ++i) {
    auto detection_array_msg = std::make_shared<vision_msgs::msg::Detection2DArray>();
    detection_array_msg->header = headers[i];

    for (const auto & detection : batch_detections[i]) {
      vision_msgs::msg::Detection2D det_msg;
      det_msg.header = headers[i];

      // Set bounding box
      det_msg.bbox.center.position.x = detection.x + detection.width / 2.0;
      det_msg.bbox.center.position.y = detection.y + detection.height / 2.0;
      det_msg.bbox.size_x = detection.width;
      det_msg.bbox.size_y = detection.height;

      // Set detection result
      vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
      hypothesis.hypothesis.class_id = std::to_string(detection.class_id);
      hypothesis.hypothesis.score = detection.confidence;
      det_msg.results.push_back(hypothesis);

      detection_array_msg->detections.push_back(det_msg);
    }

    detection_publisher_->publish(*detection_array_msg);
  }
}

void ObjectDetectionNode::publishVisualization(
  const std::vector<std::vector<Detection>> & batch_detections,
  const std::vector<cv::Mat> & images,
  const std::vector<std_msgs::msg::Header> & headers,
  const std::vector<int> & camera_ids)
{
  if (!visualization_publisher_ && !visualization_image_publisher_) {
    return;
  }

  for (size_t i = 0; i < batch_detections.size() && i < headers.size() && i < camera_ids.size(); ++i) {
    auto marker_array = createDetectionMarkers(batch_detections[i], headers[i], camera_ids[i]);
    if (visualization_publisher_) {
      visualization_publisher_->publish(marker_array);
    }

    if (visualization_image_publisher_ && i < images.size()) {
      // Draw detections on a copy of the image (avoid modifying original batch image)
      cv::Mat annotated;
      if (images[i].channels() == 3) {
        // Use shallow copy then convert to ensure we don't modify shared data
        annotated = images[i].clone();
      } else {
        cv::cvtColor(images[i], annotated, cv::COLOR_GRAY2BGR);
      }

      // Draw boxes and labels
      for (const auto & det : batch_detections[i]) {
        cv::Scalar color = getClassColor(det.class_id);
        cv::Point top_left(static_cast<int>(det.x), static_cast<int>(det.y));
        cv::Point bottom_right(static_cast<int>(det.x + det.width), static_cast<int>(det.y + det.height));
        cv::rectangle(annotated, top_left, bottom_right, color, 2);

        std::string label = det.class_name + " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        int baseline = 0;
        double font_scale = 0.5;
        int thickness = 1;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        cv::Point label_origin(top_left.x, std::max(0, top_left.y - 5));
        cv::rectangle(
          annotated,
          cv::Point(label_origin.x, label_origin.y - text_size.height - baseline),
          cv::Point(label_origin.x + text_size.width, label_origin.y + baseline),
          color,
          cv::FILLED);
        cv::putText(
          annotated,
          label,
          cv::Point(label_origin.x, label_origin.y - 2),
          cv::FONT_HERSHEY_SIMPLEX,
          font_scale,
          cv::Scalar(255, 255, 255),
          thickness);
      }

      // Convert to ROS image message
      try {
        std_msgs::msg::Header header = headers[i];
        auto img_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, annotated).toImageMsg();
        visualization_image_publisher_->publish(*img_msg);
      } catch (const cv_bridge::Exception & e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception when publishing visualization image: %s", e.what());
      }
    }
  }
}

cv::Scalar ObjectDetectionNode::getClassColor(int class_id)
{
  // Generate consistent colors for each class
  static std::vector<cv::Scalar> colors = {
    cv::Scalar(255, 0, 0),  // Red
    cv::Scalar(0, 255, 0),  // Green
    cv::Scalar(0, 0, 255),  // Blue
    cv::Scalar(255, 255, 0),  // Yellow
    cv::Scalar(255, 0, 255),  // Magenta
    cv::Scalar(0, 255, 255),  // Cyan
    cv::Scalar(128, 0, 0),  // Dark Red
    cv::Scalar(0, 128, 0),  // Dark Green
    cv::Scalar(0, 0, 128),  // Dark Blue
    cv::Scalar(255, 128, 0),  // Orange
  };

  return colors[class_id % colors.size()];
}

visualization_msgs::msg::MarkerArray ObjectDetectionNode::createDetectionMarkers(
  const std::vector<Detection> & detections, const std_msgs::msg::Header & header, int camera_id)
{
  visualization_msgs::msg::MarkerArray marker_array;

  for (size_t i = 0; i < detections.size(); ++i) {
    const auto & detection = detections[i];

    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "detections_camera_" + std::to_string(camera_id);
    marker.id = i;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;

    // Set position (assuming 2D detection, place at fixed distance)
    marker.pose.position.x = (detection.x + detection.width / 2.0) / 100.0;  // Scale down
    marker.pose.position.y = (detection.y + detection.height / 2.0) / 100.0;
    marker.pose.position.z = 1.0;  // Fixed distance

    marker.pose.orientation.w = 1.0;

    // Set scale
    marker.scale.x = detection.width / 100.0;
    marker.scale.y = detection.height / 100.0;
    marker.scale.z = 0.1;

    // Set color based on class
    auto color = getClassColor(detection.class_id);
    marker.color.r = color[2] / 255.0;  // OpenCV uses BGR
    marker.color.g = color[1] / 255.0;
    marker.color.b = color[0] / 255.0;
    marker.color.a = 0.7;

    marker.lifetime = rclcpp::Duration::from_seconds(0.5);

    marker_array.markers.push_back(marker);

    // Add text marker for class name
    visualization_msgs::msg::Marker text_marker = marker;
    text_marker.id = i + 1000;  // Offset to avoid conflicts
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.text =
      detection.class_name + " (" + std::to_string(static_cast<int>(detection.confidence * 100)) + "%)";
    text_marker.pose.position.z = 1.2;  // Slightly above the box
    text_marker.scale.z = 0.2;  // Text height
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;

    marker_array.markers.push_back(text_marker);
  }

  return marker_array;
}

void ObjectDetectionNode::start()
{
  initializeSubscribers();
  setupTimers();
  RCLCPP_INFO(this->get_logger(), "Subscribers and timers started");
}

void ObjectDetectionNode::publishPerformanceStats()
{
  if (!performance_publisher_) {
    return;
  }

  auto stats_msg = std::make_shared<std_msgs::msg::String>();
  stats_msg->data = "Total inferences: " + std::to_string(total_inferences_) +
                    ", Average time: " + std::to_string(average_inference_time_) + " ms";

  performance_publisher_->publish(*stats_msg);
}

}  // namespace deep_object_detection

// Node main function
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<deep_object_detection::ObjectDetectionNode>();

  // Start subscribers and timers after the shared_ptr is created
  node->start();

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
