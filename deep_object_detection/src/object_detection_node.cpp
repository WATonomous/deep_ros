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

#include <std_msgs/msg/string.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include "deep_object_detection/inference_interface.hpp"

#include <yaml-cpp/yaml.h>

namespace deep_object_detection
{

ObjectDetectionNode::ObjectDetectionNode(const rclcpp::NodeOptions & options)
: Node("object_detection_node", options)
, total_inferences_(0)
, average_inference_time_(0.0)
{
  initializeParameters();
  initializeInference();
  initializeSubscribers();
  initializePublishers();
  setupTimers();

  RCLCPP_INFO(this->get_logger(), "Object Detection Node initialized");
}

ObjectDetectionNode::~ObjectDetectionNode() = default;

void ObjectDetectionNode::initializeParameters()
{
  // Model parameters
  config_.model_path = this->declare_parameter<std::string>("model_path", "/path/to/model.onnx");
  config_.max_batch_size = this->declare_parameter<int>("max_batch_size", 4);
  config_.inference_rate = this->declare_parameter<double>("inference_rate", 30.0);

  // Camera parameters
  config_.camera_topics = this->declare_parameter<std::vector<std::string>>("camera_topics", {"/camera/image_raw"});
  config_.queue_size = this->declare_parameter<int>("queue_size", 10);

  // Output parameters
  config_.detection_topic = this->declare_parameter<std::string>("detection_topic", "/detections");
  config_.visualization_topic = this->declare_parameter<std::string>("visualization_topic", "/detection_markers");
  config_.enable_visualization = this->declare_parameter<bool>("enable_visualization", true);
  config_.enable_debug = this->declare_parameter<bool>("enable_debug", false);

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

  // Load class names
  auto class_names_param = this->declare_parameter<std::vector<std::string>>("class_names", std::vector<std::string>{});
  if (class_names_param.empty()) {
    // Default COCO classes for YOLOv8
    config_.inference_config.class_names = {
      "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
      "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
      "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
      "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
      "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
      "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
      "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
      "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
      "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
      "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
      "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
      "teddy bear",     "hair drier", "toothbrush"};
  } else {
    config_.inference_config.class_names = class_names_param;
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Initialized with %zu camera topics and %zu classes",
    config_.camera_topics.size(),
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
  image_transport_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
  image_subscribers_.reserve(config_.camera_topics.size());

  for (size_t i = 0; i < config_.camera_topics.size(); ++i) {
    auto subscriber = std::make_shared<image_transport::Subscriber>(image_transport_->subscribe(
      config_.camera_topics[i], config_.queue_size, [this, i](const sensor_msgs::msg::Image::ConstSharedPtr & msg) {
        this->imageCallback(msg, i);
      }));
    image_subscribers_.push_back(subscriber);

    RCLCPP_INFO(this->get_logger(), "Subscribed to camera topic: %s", config_.camera_topics[i].c_str());
  }
}

void ObjectDetectionNode::initializePublishers()
{
  detection_publisher_ =
    this->create_publisher<vision_msgs::msg::Detection2DArray>(config_.detection_topic, config_.queue_size);

  if (config_.enable_visualization) {
    visualization_publisher_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(config_.visualization_topic, config_.queue_size);
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

void ObjectDetectionNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg, int camera_id)
{
  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    addImageToBatch(cv_ptr->image, msg->header, camera_id);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
}

void ObjectDetectionNode::addImageToBatch(const cv::Mat & image, const std_msgs::msg::Header & header, int camera_id)
{
  std::lock_guard<std::mutex> lock(batch_mutex_);

  current_batch_.images.push_back(image.clone());
  current_batch_.headers.push_back(header);
  current_batch_.camera_ids.push_back(camera_id);
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
  bool timeout = elapsed.count() > 100;  // 100ms timeout

  return batch_full || timeout;
}

void ObjectDetectionNode::batchInferenceCallback()
{
  std::unique_lock<std::mutex> lock(batch_mutex_);

  if (!shouldProcessBatch()) {
    return;
  }

  // Copy current batch and clear it
  ImageBatch batch_to_process = current_batch_;
  current_batch_.clear();
  lock.unlock();

  processBatch();
}

void ObjectDetectionNode::processBatch()
{
  if (current_batch_.empty()) {
    return;
  }

  auto start_time = std::chrono::steady_clock::now();

  // Run inference
  auto batch_detections = inference_engine_->inferBatch(current_batch_.images);

  auto end_time = std::chrono::steady_clock::now();
  auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  // Update performance statistics
  total_inferences_++;
  double current_time = inference_time.count();
  average_inference_time_ = (average_inference_time_ * (total_inferences_ - 1) + current_time) / total_inferences_;

  if (config_.enable_debug) {
    RCLCPP_DEBUG(
      this->get_logger(),
      "Batch inference time: %ld ms, batch size: %zu",
      inference_time.count(),
      current_batch_.size());
    publishPerformanceStats();
  }

  // Publish results
  publishDetections(batch_detections, current_batch_.headers);

  if (config_.enable_visualization) {
    publishVisualization(batch_detections, current_batch_.images, current_batch_.headers);
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
  const std::vector<cv::Mat> & /* images */,
  const std::vector<std_msgs::msg::Header> & headers)
{
  if (!visualization_publisher_) {
    return;
  }

  for (size_t i = 0; i < batch_detections.size() && i < headers.size(); ++i) {
    auto marker_array = createDetectionMarkers(batch_detections[i], headers[i], current_batch_.camera_ids[i]);
    visualization_publisher_->publish(marker_array);
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

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
