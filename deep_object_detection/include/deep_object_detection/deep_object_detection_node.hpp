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
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#if __has_include(<deep_msgs/msg/multi_image.hpp>)
  #include <deep_msgs/msg/multi_image.hpp>
#endif

#include "deep_object_detection/backend_manager.hpp"
#include "deep_object_detection/detection_types.hpp"
#include "deep_object_detection/generic_postprocessor.hpp"
#include "deep_object_detection/image_preprocessor.hpp"

namespace deep_object_detection
{

class DeepObjectDetectionNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit DeepObjectDetectionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State & state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_shutdown(
    const rclcpp_lifecycle::State & state) override;

private:
  void declareAndReadParameters();
  void validateParameters();
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void onCompressedImage(const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg);
  void setupMultiCameraSubscriptions();
  void setupCameraSyncSubscription();
  void onMultiImage(const deep_msgs::msg::MultiImage::ConstSharedPtr & msg);
  void handleCompressedImage(const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg, int camera_id);
  void enqueueImage(cv::Mat image, const std_msgs::msg::Header & header);
  bool isCompressedTopic(const std::string & topic) const;
  std::string formatShape(const std::vector<size_t> & shape) const;
  void onBatchTimer();
  void processBatch(const std::vector<QueuedImage> & batch);
  void publishDetections(
    const std::vector<std::vector<SimpleDetection>> & batch_detections,
    const std::vector<std_msgs::msg::Header> & headers,
    const std::vector<ImageMeta> & metas);
  void loadClassNames();

  DetectionParams params_;

  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_image_sub_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr> multi_camera_subscriptions_;
#if __has_include(<deep_msgs/msg/multi_image.hpp>)
  rclcpp::Subscription<deep_msgs::msg::MultiImage>::SharedPtr multi_image_sub_;
#endif
  bool multi_camera_mode_{false};
  bool use_camera_sync_{false};
  std::string camera_sync_topic_;
  rclcpp::Publisher<Detection2DArrayMsg>::SharedPtr detection_pub_;
  rclcpp::TimerBase::SharedPtr batch_timer_;

  std::deque<QueuedImage> image_queue_;
  std::mutex queue_mutex_;
  std::atomic<bool> processing_{false};

  std::unique_ptr<ImagePreprocessor> preprocessor_;
  std::unique_ptr<GenericPostprocessor> postprocessor_;
  std::unique_ptr<BackendManager> backend_manager_;
};

std::shared_ptr<rclcpp_lifecycle::LifecycleNode> createDeepObjectDetectionNode(
  const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

}  // namespace deep_object_detection
