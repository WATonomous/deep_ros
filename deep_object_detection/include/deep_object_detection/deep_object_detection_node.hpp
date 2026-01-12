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

#include <opencv2/core/mat.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/header.hpp>

#include <deep_msgs/msg/multi_image.hpp>

#include "deep_object_detection/backend_manager.hpp"
#include "deep_object_detection/detection_types.hpp"
#include "deep_object_detection/generic_postprocessor.hpp"
#include "deep_object_detection/image_preprocessor.hpp"

namespace deep_object_detection
{

/**
 * @brief ROS2 lifecycle node for object detection using ONNX models
 *
 * This node performs object detection on images from cameras or synchronized multi-camera streams.
 * It supports:
 * - Multiple input modes: individual camera topics or synchronized MultiImage messages
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
  /**
   * @brief Declare and read all ROS2 parameters
   *
   * Reads model configuration, preprocessing/postprocessing parameters, camera topics,
   * batch settings, and backend provider settings from ROS2 parameters.
   */
  void declareAndReadParameters();

  /**
   * @brief Setup subscriptions to individual camera compressed image topics
   *
   * Creates one subscription per camera topic in params_.camera_topics.
   * Each subscription calls handleCompressedImage() with the camera index.
   */
  void setupMultiCameraSubscriptions();

  /**
   * @brief Setup subscription to synchronized MultiImage topic
   *
   * Creates a single subscription to camera_sync_topic_ that receives MultiImage
   * messages containing synchronized compressed images from multiple cameras.
   */
  void setupCameraSyncSubscription();

  /**
   * @brief Handle incoming MultiImage message with synchronized images
   * @param msg MultiImage message containing multiple compressed images
   *
   * Extracts each compressed image from the MultiImage and processes them
   * through handleCompressedImage() with sequential camera IDs.
   */
  void onMultiImage(const deep_msgs::msg::MultiImage::ConstSharedPtr & msg);

  /**
   * @brief Handle incoming compressed image from a camera
   * @param msg Compressed image message
   * @param camera_id Camera identifier (index for multi-camera, or from MultiImage)
   *
   * Decodes the compressed image, enqueues it for batch processing.
   * Handles decode failures according to decode_failure_policy.
   */
  void handleCompressedImage(const sensor_msgs::msg::CompressedImage & msg, int camera_id);

  /**
   * @brief Add image to processing queue
   * @param image Decoded BGR image (OpenCV Mat)
   * @param header ROS message header with timestamp and frame_id
   *
   * Thread-safe enqueueing. Applies queue_overflow_policy if queue is full.
   * Tracks first image timestamp for batch timeout calculation.
   */
  void enqueueImage(cv::Mat image, const std_msgs::msg::Header & header);

  /**
   * @brief Format tensor shape vector as string for logging
   * @param shape Vector of dimension sizes
   * @return Comma-separated string representation (e.g., "1, 3, 640, 640")
   */
  std::string formatShape(const std::vector<size_t> & shape) const;

  /**
   * @brief Timer callback for batch processing
   *
   * Called periodically (every 5ms) to check if batch should be processed.
   * Processes batch if:
   * - Queue size >= min_batch_size, OR
   * - max_batch_latency_ms exceeded and queue not empty
   * Extracts up to max_batch_size images and calls processBatch().
   */
  void onBatchTimer();

  /**
   * @brief Process a batch of images through inference pipeline
   * @param batch Vector of queued images to process
   *
   * For each image: preprocess -> inference -> postprocess -> publish detections.
   * Handles multi-output models if configured. Publishes Detection2DArray messages.
   */
  void processBatch(const std::vector<QueuedImage> & batch);

  /**
   * @brief Publish detection results for a batch
   * @param batch_detections Detections for each image in batch
   * @param headers Message headers for each image (for frame_id and timestamp)
   * @param metas Image metadata for coordinate transformation
   *
   * Creates and publishes Detection2DArray message for each image with its detections.
   */
  void publishDetections(
    const std::vector<std::vector<SimpleDetection>> & batch_detections,
    const std::vector<std_msgs::msg::Header> & headers,
    const std::vector<ImageMeta> & metas);

  /**
   * @brief Load class names from file
   *
   * Reads class names from params_.model_metadata.class_names_file (one per line).
   * Stores in params_.class_names for use in postprocessing and message publishing.
   */
  void loadClassNames();

  /**
   * @brief Stop all subscriptions and cancel batch timer
   *
   * Clears all camera subscriptions, resets MultiImage subscription,
   * cancels batch timer, and clears image queue. Used in deactivate/cleanup/shutdown.
   */
  void stopSubscriptionsAndTimer();

  DetectionParams params_;  ///< All node configuration parameters

  std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr>
    multi_camera_subscriptions_;  ///< Subscriptions for individual camera topics
  rclcpp::Subscription<deep_msgs::msg::MultiImage>::SharedPtr
    multi_image_sub_;  ///< Subscription for synchronized MultiImage messages
  bool use_camera_sync_{false};  ///< Whether to use MultiImage sync mode or individual topics
  std::string camera_sync_topic_;  ///< Topic name for MultiImage messages
  rclcpp::Publisher<Detection2DArrayMsg>::SharedPtr detection_pub_;  ///< Publisher for detection results
  rclcpp::TimerBase::SharedPtr batch_timer_;  ///< Timer for periodic batch processing checks

  std::deque<QueuedImage> image_queue_;  ///< Queue of images waiting for batch processing
  std::mutex queue_mutex_;  ///< Mutex protecting image_queue_ and first_image_timestamp_
  std::atomic<bool> processing_{false};  ///< Flag to prevent concurrent batch processing
  rclcpp::Time first_image_timestamp_;  ///< Timestamp of oldest image in queue (for batch timeout)

  std::unique_ptr<ImagePreprocessor> preprocessor_;  ///< Image preprocessing (resize, normalize, etc.)
  std::unique_ptr<GenericPostprocessor> postprocessor_;  ///< Detection postprocessing (NMS, decode, etc.)
  std::unique_ptr<BackendManager> backend_manager_;  ///< Backend plugin manager (CPU/CUDA/TensorRT)
};

/**
 * @brief Factory function to create DeepObjectDetectionNode instance
 * @param options Node options for ROS2 configuration
 * @return Shared pointer to lifecycle node
 *
 * Used by rclcpp_components for component loading.
 */
std::shared_ptr<rclcpp_lifecycle::LifecycleNode> createDeepObjectDetectionNode(
  const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

}  // namespace deep_object_detection
