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

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <image_transport/subscriber.hpp>
#include <opencv2/core/mat.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "deep_yolo_inference/backend_manager.hpp"
#include "deep_yolo_inference/detection_msg_alias.hpp"
#include "deep_yolo_inference/processing.hpp"
#include "deep_yolo_inference/yolo_types.hpp"

namespace deep_yolo_inference
{

class YoloInferenceNode : public rclcpp::Node
{
public:
  explicit YoloInferenceNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void declareAndReadParameters();
  void validateParameters();
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void onCompressedImage(const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg);
  void setupMultiCameraSubscriptions();
  void handleCompressedImage(const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg, int camera_id);
  void enqueueImage(cv::Mat image, const std_msgs::msg::Header & header);
  size_t queueLimit() const;
  bool isCompressedTopic(const std::string & topic) const;
  std::string formatShape(const std::vector<size_t> & shape) const;
  void onBatchTimer();
  void processBatch(const std::vector<QueuedImage> & batch);
  void publishDetections(
    const std::vector<std::vector<SimpleDetection>> & batch_detections,
    const std::vector<std_msgs::msg::Header> & headers,
    const std::vector<ImageMeta> & metas);
  void loadClassNames();

  YoloParams params_;

  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_image_sub_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr> multi_camera_subscriptions_;
  bool multi_camera_mode_{false};
  rclcpp::Publisher<Detection2DArrayMsg>::SharedPtr detection_pub_;
  rclcpp::TimerBase::SharedPtr batch_timer_;

  std::deque<QueuedImage> image_queue_;
  std::mutex queue_mutex_;
  std::atomic<bool> processing_{false};
  std::vector<std::string> class_names_;

  std::unique_ptr<ImagePreprocessor> preprocessor_;
  std::unique_ptr<Postprocessor> postprocessor_;
  std::unique_ptr<BackendManager> backend_manager_;
};

/**
 * @brief Factory function to create the YOLO inference node
 */
std::shared_ptr<rclcpp::Node> createYoloInferenceNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

}  // namespace deep_yolo_inference
