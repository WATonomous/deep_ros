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

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <deep_msgs/msg/multi_image.hpp>
#include <opencv2/core/mat.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/header.hpp>

#include "deep_object_detection/backend_manager.hpp"
#include "deep_object_detection/detection_types.hpp"
#include "deep_object_detection/generic_postprocessor.hpp"
#include "deep_object_detection/image_preprocessor.hpp"

namespace deep_object_detection
{

/**
 * @brief ROS2 lifecycle node for object detection using ONNX models
 *
 * This node performs object detection on synchronized multi-camera streams via MultiImage messages.
 * It supports:
 * - MultiImage input: synchronized compressed images from multiple cameras
 * - Batch processing: groups images for efficient inference
 * - Multiple backends: CPU, CUDA, or TensorRT execution providers
 * - Configurable preprocessing: resizing, normalization, color format conversion
 * - Flexible postprocessing: NMS, score thresholds, multiple output formats
 *
 * The node follows the ROS2 lifecycle pattern:
 * - unconfigured -> configuring -> inactive -> activating -> active
 * - Can be deactivated/activated without full cleanup
 * - Subscriptions and publishers are only active when the node is in the active state
 */
class DeepObjectDetectionNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  /**
   * @brief Construct the object detection node
   * @param options Node options for ROS2 configuration
   */
  explicit DeepObjectDetectionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  /**
   * @brief Configure the node: load model, initialize backends, setup postprocessor
   * @return SUCCESS if configuration succeeds, FAILURE otherwise
   *
   * Loads class names, initializes image preprocessor, backend manager, and postprocessor.
   * Detects model output shape and configures layout. Does not start subscriptions.
   */
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State &) override;

  /**
   * @brief Activate the node: start subscriptions and batch processing timer
   * @return SUCCESS if activation succeeds, FAILURE otherwise
   *
   * Creates subscriptions (either MultiImage or individual camera topics based on configuration),
   * starts the batch processing timer, and activates the detection publisher.
   */
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State &) override;

  /**
   * @brief Deactivate the node: stop subscriptions and timer
   * @return SUCCESS
   *
   * Stops all subscriptions, cancels the batch timer, and deactivates the publisher.
   * Node can be reactivated without full cleanup.
   */
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State &) override;

  /**
   * @brief Cleanup the node: release all resources
   * @return SUCCESS
   *
   * Releases backend manager, preprocessor, postprocessor, and clears all data.
   * Node must be reconfigured before reactivation.
   */
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State &) override;

  /**
   * @brief Shutdown the node: final cleanup
   * @return SUCCESS
   *
   * Performs final cleanup. Node cannot be reactivated after shutdown.
   */
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_shutdown(
    const rclcpp_lifecycle::State &) override;

private:
  void declareAndReadParameters();
  void setupSubscription();
  void onMultiImage(const deep_msgs::msg::MultiImage::ConstSharedPtr & msg);
  void handleCompressedImage(const sensor_msgs::msg::CompressedImage & msg);
  void enqueueImage(cv::Mat image, const std_msgs::msg::Header & header);
  void onBatchTimer();
  void processBatch(const std::vector<QueuedImage> & batch);
  void publishDetections(
    const std::vector<std::vector<SimpleDetection>> & batch_detections,
    const std::vector<std_msgs::msg::Header> & headers,
    const std::vector<ImageMeta> & metas);
  void loadClassNames();
  void cleanupPartialConfiguration();
  void cleanupAllResources();
  void stopSubscriptionsAndTimer();

  DetectionParams params_;

  rclcpp::Subscription<deep_msgs::msg::MultiImage>::SharedPtr multi_image_sub_;
  std::string input_topic_;
  rclcpp_lifecycle::LifecyclePublisher<Detection2DArrayMsg>::SharedPtr detection_pub_;
  rclcpp::TimerBase::SharedPtr batch_timer_;

  std::deque<QueuedImage> image_queue_;
  std::mutex queue_mutex_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;

  size_t dropped_images_count_;

  std::unique_ptr<ImagePreprocessor> preprocessor_;
  std::unique_ptr<GenericPostprocessor> postprocessor_;
  std::unique_ptr<BackendManager> backend_manager_;
};

std::shared_ptr<rclcpp_lifecycle::LifecycleNode> createDeepObjectDetectionNode(
  const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

}  // namespace deep_object_detection
