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

#ifndef CAMERA_SYNC__MULTI_CAMERA_SYNC_NODE_HPP_
#define CAMERA_SYNC__MULTI_CAMERA_SYNC_NODE_HPP_

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

// Conditional lifecycle node support for Rolling and newer
// USE_LIFECYCLE_NODE is defined by CMake based on ROS distribution
#ifndef USE_LIFECYCLE_NODE
  #define USE_LIFECYCLE_NODE 0  // Default fallback for intellisense
#endif

#if USE_LIFECYCLE_NODE
  #include <lifecycle_msgs/msg/state.hpp>
  #include <rclcpp_lifecycle/lifecycle_node.hpp>
#endif

#include "deep_msgs/msg/multi_image.hpp"
#include "deep_msgs/msg/multi_image_raw.hpp"

namespace camera_sync
{

/**
 * @brief Node that synchronizes N camera image messages using message filters
 *
 * This node can handle both raw images (sensor_msgs/Image) and compressed images
 * (sensor_msgs/CompressedImage) and synchronize them based on their timestamps.
 * Supports 2-6 cameras with a compact implementation.
 */
#if USE_LIFECYCLE_NODE
class MultiCameraSyncNode : public rclcpp_lifecycle::LifecycleNode
#else
class MultiCameraSyncNode : public rclcpp::Node
#endif
{
public:
  explicit MultiCameraSyncNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~MultiCameraSyncNode() = default;

#if USE_LIFECYCLE_NODE
  // Lifecycle node callbacks
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
#endif

private:
  // Message types
  using ImageMsg = sensor_msgs::msg::Image;
  using CompressedImageMsg = sensor_msgs::msg::CompressedImage;

  // Subscriber types
#if USE_LIFECYCLE_NODE
  using ImageSubscriber = message_filters::Subscriber<ImageMsg, rclcpp_lifecycle::LifecycleNode>;
  using CompressedImageSubscriber = message_filters::Subscriber<CompressedImageMsg, rclcpp_lifecycle::LifecycleNode>;
#else
  using ImageSubscriber = message_filters::Subscriber<ImageMsg>;
  using CompressedImageSubscriber = message_filters::Subscriber<CompressedImageMsg>;
#endif

  // Sync policies for raw images
  using ImageSyncPolicy2 = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;
  using ImageSyncPolicy3 = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg, ImageMsg>;
  using ImageSyncPolicy4 = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg, ImageMsg, ImageMsg>;
  using ImageSyncPolicy5 =
    message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg, ImageMsg, ImageMsg, ImageMsg>;
  using ImageSyncPolicy6 =
    message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg, ImageMsg, ImageMsg, ImageMsg, ImageMsg>;

  // Sync policies for compressed images
  using CompressedSyncPolicy2 = message_filters::sync_policies::ApproximateTime<CompressedImageMsg, CompressedImageMsg>;
  using CompressedSyncPolicy3 =
    message_filters::sync_policies::ApproximateTime<CompressedImageMsg, CompressedImageMsg, CompressedImageMsg>;
  using CompressedSyncPolicy4 = message_filters::sync_policies::
    ApproximateTime<CompressedImageMsg, CompressedImageMsg, CompressedImageMsg, CompressedImageMsg>;
  using CompressedSyncPolicy5 = message_filters::sync_policies::
    ApproximateTime<CompressedImageMsg, CompressedImageMsg, CompressedImageMsg, CompressedImageMsg, CompressedImageMsg>;
  using CompressedSyncPolicy6 = message_filters::sync_policies::ApproximateTime<
    CompressedImageMsg,
    CompressedImageMsg,
    CompressedImageMsg,
    CompressedImageMsg,
    CompressedImageMsg,
    CompressedImageMsg>;

  /**
   * @brief Initialize the node parameters
   */
  void initializeParameters();

  /**
   * @brief Setup subscribers and synchronizers
   */
  void setupSynchronization();

  /**
   * @brief Setup raw image synchronization
   */
  void setupRawSync(size_t num_cameras);

  /**
   * @brief Setup compressed image synchronization
   */
  void setupCompressedSync(size_t num_cameras);

  /**
   * @brief Callback functions for synchronized raw images
   */
  void syncCallback2Raw(const ImageMsg::ConstSharedPtr & img1, const ImageMsg::ConstSharedPtr & img2);
  void syncCallback3Raw(
    const ImageMsg::ConstSharedPtr & img1,
    const ImageMsg::ConstSharedPtr & img2,
    const ImageMsg::ConstSharedPtr & img3);
  void syncCallback4Raw(
    const ImageMsg::ConstSharedPtr & img1,
    const ImageMsg::ConstSharedPtr & img2,
    const ImageMsg::ConstSharedPtr & img3,
    const ImageMsg::ConstSharedPtr & img4);
  void syncCallback5Raw(
    const ImageMsg::ConstSharedPtr & img1,
    const ImageMsg::ConstSharedPtr & img2,
    const ImageMsg::ConstSharedPtr & img3,
    const ImageMsg::ConstSharedPtr & img4,
    const ImageMsg::ConstSharedPtr & img5);
  void syncCallback6Raw(
    const ImageMsg::ConstSharedPtr & img1,
    const ImageMsg::ConstSharedPtr & img2,
    const ImageMsg::ConstSharedPtr & img3,
    const ImageMsg::ConstSharedPtr & img4,
    const ImageMsg::ConstSharedPtr & img5,
    const ImageMsg::ConstSharedPtr & img6);

  /**
   * @brief Callback functions for synchronized compressed images
   */
  void syncCallback2Compressed(
    const CompressedImageMsg::ConstSharedPtr & img1, const CompressedImageMsg::ConstSharedPtr & img2);
  void syncCallback3Compressed(
    const CompressedImageMsg::ConstSharedPtr & img1,
    const CompressedImageMsg::ConstSharedPtr & img2,
    const CompressedImageMsg::ConstSharedPtr & img3);
  void syncCallback4Compressed(
    const CompressedImageMsg::ConstSharedPtr & img1,
    const CompressedImageMsg::ConstSharedPtr & img2,
    const CompressedImageMsg::ConstSharedPtr & img3,
    const CompressedImageMsg::ConstSharedPtr & img4);
  void syncCallback5Compressed(
    const CompressedImageMsg::ConstSharedPtr & img1,
    const CompressedImageMsg::ConstSharedPtr & img2,
    const CompressedImageMsg::ConstSharedPtr & img3,
    const CompressedImageMsg::ConstSharedPtr & img4,
    const CompressedImageMsg::ConstSharedPtr & img5);
  void syncCallback6Compressed(
    const CompressedImageMsg::ConstSharedPtr & img1,
    const CompressedImageMsg::ConstSharedPtr & img2,
    const CompressedImageMsg::ConstSharedPtr & img3,
    const CompressedImageMsg::ConstSharedPtr & img4,
    const CompressedImageMsg::ConstSharedPtr & img5,
    const CompressedImageMsg::ConstSharedPtr & img6);

  /**
   * @brief Process synchronized images (statistics and custom logic)
   * @param timestamps Vector of synchronized timestamps from all cameras
   */
  void processSynchronizedImages(const std::vector<rclcpp::Time> & timestamps);

  // Parameters
  std::vector<std::string> camera_topics_;
  std::vector<std::string> camera_names_;
  std::vector<sensor_msgs::msg::Image::ConstSharedPtr> image_array;
  bool use_compressed_;
  double sync_tolerance_ms_;
  int queue_size_;
  bool publish_sync_info_;

  // Subscribers
  std::vector<std::unique_ptr<ImageSubscriber>> image_subscribers_;
  std::vector<std::unique_ptr<CompressedImageSubscriber>> compressed_subscribers_;

  // Synchronizers for raw images
  std::unique_ptr<message_filters::Synchronizer<ImageSyncPolicy2>> sync2_raw_;
  std::unique_ptr<message_filters::Synchronizer<ImageSyncPolicy3>> sync3_raw_;
  std::unique_ptr<message_filters::Synchronizer<ImageSyncPolicy4>> sync4_raw_;
  std::unique_ptr<message_filters::Synchronizer<ImageSyncPolicy5>> sync5_raw_;
  std::unique_ptr<message_filters::Synchronizer<ImageSyncPolicy6>> sync6_raw_;

  // Synchronizers for compressed images
  std::unique_ptr<message_filters::Synchronizer<CompressedSyncPolicy2>> sync2_compressed_;
  std::unique_ptr<message_filters::Synchronizer<CompressedSyncPolicy3>> sync3_compressed_;
  std::unique_ptr<message_filters::Synchronizer<CompressedSyncPolicy4>> sync4_compressed_;
  std::unique_ptr<message_filters::Synchronizer<CompressedSyncPolicy5>> sync5_compressed_;
  std::unique_ptr<message_filters::Synchronizer<CompressedSyncPolicy6>> sync6_compressed_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr sync_info_pub_;

  // Publishers for multi-image messages
  rclcpp::Publisher<deep_msgs::msg::MultiImageRaw>::SharedPtr multi_image_raw_pub_;
  rclcpp::Publisher<deep_msgs::msg::MultiImage>::SharedPtr multi_image_compressed_pub_;

  // Statistics
  int64_t sync_count_;
  rclcpp::Time last_sync_time_;
  std::chrono::steady_clock::time_point start_time_;
};

}  // namespace camera_sync

#endif  // camera_sync__MULTI_CAMERA_SYNC_NODE_HPP_
