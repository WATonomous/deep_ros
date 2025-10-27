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

#pragma once

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <memory>
#include <string>
#include <vector>

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "deep_object_detection/inference_interface.hpp"

namespace deep_object_detection
{

class ObjectDetectionNode : public rclcpp::Node
{
public:
  explicit ObjectDetectionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~ObjectDetectionNode();

private:
  // Configuration parameters
  struct NodeConfig
  {
    std::string model_path;
    std::vector<std::string> camera_topics;
    std::string detection_topic;
    std::string visualization_topic;
    int max_batch_size;
    int queue_size;
    double inference_rate;
    bool enable_visualization;
    bool enable_debug;
    InferenceConfig inference_config;
  };

  // Core methods
  void initializeParameters();
  void initializeSubscribers();
  void initializePublishers();
  void initializeInference();
  void setupTimers();

  // Callback methods
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg, int camera_id);
  void batchInferenceCallback();
  void publishDetections(
    const std::vector<std::vector<Detection>> & batch_detections, const std::vector<std_msgs::msg::Header> & headers);
  void publishVisualization(
    const std::vector<std::vector<Detection>> & batch_detections,
    const std::vector<cv::Mat> & images,
    const std::vector<std_msgs::msg::Header> & headers);

  // Image batch management
  struct ImageBatch
  {
    std::vector<cv::Mat> images;
    std::vector<std_msgs::msg::Header> headers;
    std::vector<int> camera_ids;
    std::chrono::steady_clock::time_point timestamp;

    void clear()
    {
      images.clear();
      headers.clear();
      camera_ids.clear();
    }

    bool empty() const
    {
      return images.empty();
    }

    size_t size() const
    {
      return images.size();
    }
  };

  void addImageToBatch(const cv::Mat & image, const std_msgs::msg::Header & header, int camera_id);
  bool shouldProcessBatch() const;
  void processBatch();

  // Utility methods
  cv::Scalar getClassColor(int class_id);
  visualization_msgs::msg::MarkerArray createDetectionMarkers(
    const std::vector<Detection> & detections, const std_msgs::msg::Header & header, int camera_id);

  // ROS2 components
  std::vector<std::shared_ptr<image_transport::Subscriber>> image_subscribers_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr visualization_publisher_;
  rclcpp::TimerBase::SharedPtr batch_timer_;

  // Image transport
  std::shared_ptr<image_transport::ImageTransport> image_transport_;

  // Inference engine
  std::unique_ptr<InferenceInterface> inference_engine_;

  // Configuration and state
  NodeConfig config_;
  ImageBatch current_batch_;
  std::mutex batch_mutex_;

  // Performance monitoring
  std::chrono::steady_clock::time_point last_inference_time_;
  size_t total_inferences_;
  double average_inference_time_;

  // Publishers for performance monitoring
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr performance_publisher_;
  void publishPerformanceStats();
};

}  // namespace deep_object_detection
