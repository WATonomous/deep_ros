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

#include <deep_core/deep_node_base.hpp>
#include <deep_core/types/tensor.hpp>
#include <deep_msgs/msg/multi_image.hpp>
#include <deep_msgs/msg/multi_image_raw.hpp>
#include <opencv2/core/mat.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/image_marker.hpp>

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
class DeepObjectDetectionNode : public deep_ros::DeepNodeBase
{
public:
  /**
   * @brief Construct the object detection node
   * @param options Node options for ROS2 configuration
   */
  explicit DeepObjectDetectionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  /**
   * @brief Destructor
   */
  ~DeepObjectDetectionNode() override = default;

protected:
  /**
   * @brief Configure the node: setup postprocessor
   * @return SUCCESS if configuration succeeds, FAILURE otherwise
   *
   * Loads class names, initializes image preprocessor, and postprocessor.
   * Detects model output shape and configures layout. Does not start subscriptions.
   * Plugin and model loading are handled by DeepNodeBase.
   */
  deep_ros::CallbackReturn on_configure_impl(const rclcpp_lifecycle::State & /*state*/) override;

  /**
   * @brief Activate the node: start subscriptions
   * @return SUCCESS if activation succeeds, FAILURE otherwise
   *
   * Creates MultiImage subscription and activates the detection publisher.
   */
  deep_ros::CallbackReturn on_activate_impl(const rclcpp_lifecycle::State & /*state*/) override;

  /**
   * @brief Deactivate the node: stop subscriptions
   * @return SUCCESS
   *
   * Stops all subscriptions and deactivates the publisher.
   * Node can be reactivated without full cleanup.
   */
  deep_ros::CallbackReturn on_deactivate_impl(const rclcpp_lifecycle::State & /*state*/) override;

  /**
   * @brief Cleanup the node: release all resources
   * @return SUCCESS
   *
   * Releases preprocessor, postprocessor, and clears all data.
   * Plugin and model cleanup are handled by DeepNodeBase.
   * Node must be reconfigured before reactivation.
   */
  deep_ros::CallbackReturn on_cleanup_impl(const rclcpp_lifecycle::State & /*state*/) override;

  /**
   * @brief Shutdown the node: final cleanup
   * @return SUCCESS
   *
   * Performs final cleanup. Plugin and model cleanup are handled by DeepNodeBase.
   * Node cannot be reactivated after shutdown.
   */
  deep_ros::CallbackReturn on_shutdown_impl(const rclcpp_lifecycle::State & /*state*/) override;

private:
  /**
   * @brief Declare and load all ROS parameters from the parameter server
   *
   * Declares all parameters with defaults and loads them into params_ struct.
   * All parameters must exist in the YAML config file.
   */
  void declareParameters();

  /**
   * @brief Setup the MultiImage subscription
   *
   * Creates subscriptions to both compressed and uncompressed MultiImage topics with best_effort QoS.
   * Subscriptions are only active when node is in active state.
   */
  void setupSubscription();

  /**
   * @brief Callback for MultiImage messages (compressed)
   * @param msg Shared pointer to MultiImage message containing synchronized compressed images
   *
   * Converts compressed images to cv::Mat and processes them.
   */
  void onMultiImage(const deep_msgs::msg::MultiImage::ConstSharedPtr & msg);

  /**
   * @brief Callback for MultiImageRaw messages (uncompressed)
   * @param msg Shared pointer to MultiImageRaw message containing synchronized uncompressed images
   *
   * Converts uncompressed images to cv::Mat and processes them.
   */
  void onMultiImageRaw(const deep_msgs::msg::MultiImageRaw::ConstSharedPtr & msg);

  /**
   * @brief Process images through the inference pipeline
   * @param images Vector of cv::Mat images to process
   * @param headers ROS headers for each image (for timestamp and frame_id)
   *
   * Runs preprocessing, inference, and postprocessing.
   * Publishes detection results for each image.
   */
  void processImages(const std::vector<cv::Mat> & images, const std::vector<std_msgs::msg::Header> & headers);

  /**
   * @brief Convert CompressedImage to cv::Mat
   * @param compressed_img Compressed image message
   * @return Decoded cv::Mat image, or empty Mat if decoding fails
   */
  cv::Mat decodeCompressedImage(const sensor_msgs::msg::CompressedImage & compressed_img);

  /**
   * @brief Convert Image to cv::Mat
   * @param img Uncompressed image message
   * @return cv::Mat image, or empty Mat if conversion fails
   */
  cv::Mat decodeImage(const sensor_msgs::msg::Image & img);

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
   * Reads class names from class_names_path parameter (one per line) and stores them
   * in class_names_. If file doesn't exist or is empty, class_names_
   * remains empty and class IDs will be used in output messages.
   */
  void loadClassNames();

  /**
   * @brief Stop all subscriptions
   *
   * Called during on_deactivate() to stop processing without full cleanup.
   * Node can be reactivated without reconfiguration.
   */
  void stopSubscriptions();

  /// Configuration parameters loaded from ROS parameter server
  DetectionParams params_;
  /// Class names loaded from file
  std::vector<std::string> class_names_;

  /// Subscription to MultiImage topic (synchronized multi-camera compressed input)
  rclcpp::Subscription<deep_msgs::msg::MultiImage>::SharedPtr multi_image_sub_;
  /// Subscription to MultiImageRaw topic (synchronized multi-camera uncompressed input)
  rclcpp::Subscription<deep_msgs::msg::MultiImageRaw>::SharedPtr multi_image_raw_sub_;
  /// Input topic name (can be set via parameter or remapping)
  std::string input_topic_;
  /// Input topic name for uncompressed images (can be set via remapping)
  std::string input_topic_raw_;
  /// Whether to use compressed images (true) or uncompressed images (false)
  bool use_compressed_images_;
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
};

/**
 * @brief Factory function to create a DeepObjectDetectionNode instance
 * @param options ROS2 node options for configuration
 * @return Shared pointer to lifecycle node instance
 *
 * Used by rclcpp_components for composable node loading.
 * Allows the node to be loaded as a component in a component container.
 */
std::shared_ptr<deep_ros::DeepNodeBase> createDeepObjectDetectionNode(
  const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

}  // namespace deep_object_detection
