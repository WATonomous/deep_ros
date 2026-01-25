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

/**
 * @file deep_object_detection_node.hpp
 * @brief ROS 2 lifecycle node for object detection using ONNX models
 *
 * This header defines the main DeepObjectDetectionNode class which:
 * - Subscribes to MultiImage messages (synchronized multi-camera input)
 * - Processes MultiImage messages directly without additional batching
 * - Runs preprocessing, inference, and postprocessing
 * - Publishes Detection2DArray messages with bounding boxes and scores
 * - Manages lifecycle states (configure, activate, deactivate, cleanup, shutdown)
 */

#pragma once

#include <memory>
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
#include <visualization_msgs/msg/image_marker.hpp>

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
 * - Direct processing: processes MultiImage messages immediately without queuing
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
   * @brief Activate the node: start subscriptions
   * @return SUCCESS if activation succeeds, FAILURE otherwise
   *
   * Creates MultiImage subscription and activates the detection publisher.
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
  /**
   * @brief Declare and read all ROS parameters from the parameter server
   *
   * Reads configuration for model, preprocessing, postprocessing, execution provider,
   * and topic names. Validates required parameters and sets defaults.
   */
  void declareAndReadParameters();

  /**
   * @brief Setup the MultiImage subscription
   *
   * Creates a subscription to the input_topic_ with best_effort QoS.
   * Subscription is only active when node is in active state.
   */
  void setupSubscription();

  /**
   * @brief Callback for MultiImage messages
   * @param msg Shared pointer to MultiImage message containing synchronized compressed images
   *
   * Decodes compressed images in the MultiImage message and processes them directly.
   * Failed decodes are dropped with a warning.
   */
  void onMultiImage(const deep_msgs::msg::MultiImage::ConstSharedPtr & msg);

  /**
   * @brief Process a MultiImage message through the inference pipeline
   * @param msg MultiImage message containing synchronized compressed images
   *
   * Decodes images, runs preprocessing, inference, and postprocessing.
   * Publishes detection results for each image in the MultiImage.
   */
  void processMultiImage(const deep_msgs::msg::MultiImage::ConstSharedPtr & msg);

  /**
   * @brief Publish detection results for a batch
   * @param batch_detections Detections for each image in the batch
   * @param headers ROS headers for each image (for timestamp and frame_id)
   * @param metas Image metadata for coordinate transformations
   *
   * Converts SimpleDetection objects to Detection2DArray messages and publishes
   * them. Each image in the batch gets its own Detection2DArray message.
   */
  void publishDetections(
    const std::vector<std::vector<SimpleDetection>> & batch_detections,
    const std::vector<std_msgs::msg::Header> & headers,
    const std::vector<ImageMeta> & metas);

  /**
   * @brief Convert detections to ImageMarker annotations for Foxglove visualization
   * @param header ROS header for the message
   * @param detections Vector of detections for a single image
   * @return ImageMarker message with bounding box and label annotations
   *
   * Converts SimpleDetection objects to visualization_msgs::msg::ImageMarker format
   * for rendering in Foxglove and other visualization tools.
   */
  visualization_msgs::msg::ImageMarker detectionsToImageMarker(
    const std_msgs::msg::Header & header, const std::vector<SimpleDetection> & detections) const;

  /**
   * @brief Load class names from file
   *
   * Reads class names from class_names_path (one per line) and stores them
   * in params_.class_names. If file doesn't exist or is empty, class_names
   * remains empty and class IDs will be used in output messages.
   */
  void loadClassNames();

  /**
   * @brief Cleanup resources if configuration partially fails
   *
   * Releases resources that were allocated during on_configure() if an error
   * occurs partway through configuration. Ensures no resource leaks.
   */
  void cleanupPartialConfiguration();

  /**
   * @brief Cleanup all node resources
   *
   * Releases backend_manager_, preprocessor_, postprocessor_, and clears
   * all subscriptions. Called during on_cleanup() and on_shutdown().
   */
  void cleanupAllResources();

  /**
   * @brief Stop all subscriptions
   *
   * Called during on_deactivate() to stop processing without full cleanup.
   * Node can be reactivated without reconfiguration.
   */
  void stopSubscriptions();

  /// Configuration parameters loaded from ROS parameter server
  DetectionParams params_;

  /// Subscription to MultiImage topic (synchronized multi-camera input)
  rclcpp::Subscription<deep_msgs::msg::MultiImage>::SharedPtr multi_image_sub_;
  /// Input topic name (can be set via parameter or remapping)
  std::string input_topic_;
  /// Publisher for Detection2DArray messages
  rclcpp_lifecycle::LifecyclePublisher<Detection2DArrayMsg>::SharedPtr detection_pub_;
  /// Publisher for ImageMarker annotations (for Foxglove visualization)
  rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::ImageMarker>::SharedPtr image_marker_pub_;
  /// Output annotations topic name for ImageMarker messages
  std::string output_annotations_topic_;
  /// Callback group for subscription (allows concurrent execution)
  rclcpp::CallbackGroup::SharedPtr callback_group_;

  /// Image preprocessor (resize, normalize, color conversion)
  std::unique_ptr<ImagePreprocessor> preprocessor_;
  /// Postprocessor (NMS, coordinate transformation, message formatting)
  std::unique_ptr<GenericPostprocessor> postprocessor_;
  /// Backend manager (handles ONNX Runtime plugin loading and inference)
  std::unique_ptr<BackendManager> backend_manager_;
};

/**
 * @brief Factory function to create a DeepObjectDetectionNode instance
 * @param options ROS2 node options for configuration
 * @return Shared pointer to lifecycle node instance
 *
 * Used by rclcpp_components for composable node loading.
 * Allows the node to be loaded as a component in a component container.
 */
std::shared_ptr<rclcpp_lifecycle::LifecycleNode> createDeepObjectDetectionNode(
  const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

}  // namespace deep_object_detection
