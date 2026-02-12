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

#if __has_include(<cv_bridge/cv_bridge.hpp>)
  #include <cv_bridge/cv_bridge.hpp>
#else
  #include <cv_bridge/cv_bridge.h>
#endif

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

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
#include "deep_msgs/msg/multi_image_compressed.hpp"

namespace camera_sync
{

/**
 * @brief Node that synchronizes N camera image messages using manual timestamp-based buffering
 *
 * This node can handle both raw images (sensor_msgs/Image) and compressed images
 * (sensor_msgs/CompressedImage) and synchronize them based on their timestamps.
 * Supports 2-12 cameras with a unified manual synchronization implementation.
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
  using ImageSubscriber = rclcpp::Subscription<ImageMsg>;
  using CompressedImageSubscriber = rclcpp::Subscription<CompressedImageMsg>;

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
   * @brief Process synchronized images (statistics and custom logic)
   * @param timestamps Vector of synchronized timestamps from all cameras
   */
  void processSynchronizedImages(const std::vector<rclcpp::Time> & timestamps);

  // Parameters
  std::vector<std::string> camera_topics_;
  std::vector<std::string> camera_names_;
  bool use_compressed_;
  double sync_tolerance_ms_;
  int queue_size_;

  // Subscribers
  std::vector<std::shared_ptr<ImageSubscriber>> image_subscribers_;
  std::vector<std::shared_ptr<CompressedImageSubscriber>> compressed_subscribers_;

  // Publishers for multi-image messages
  rclcpp::Publisher<deep_msgs::msg::MultiImage>::SharedPtr multi_image_raw_pub_;
  rclcpp::Publisher<deep_msgs::msg::MultiImageCompressed>::SharedPtr multi_image_compressed_pub_;

  // Statistics
  int64_t sync_count_;
  rclcpp::Time last_sync_time_;
  std::chrono::steady_clock::time_point start_time_;

  struct ImageBuffer
  {
    std::map<uint64_t, ImageMsg::ConstSharedPtr> buffer;
    std::shared_ptr<std::mutex> mutex;

    ImageBuffer()
    : mutex(std::make_shared<std::mutex>())
    {}
  };

  struct CompressedImageBuffer
  {
    std::map<uint64_t, CompressedImageMsg::ConstSharedPtr> buffer;
    std::shared_ptr<std::mutex> mutex;

    CompressedImageBuffer()
    : mutex(std::make_shared<std::mutex>())
    {}
  };

  std::vector<ImageBuffer> raw_image_buffers_;
  std::vector<CompressedImageBuffer> compressed_image_buffers_;

  void handleRawImageCallback(size_t camera_idx, const ImageMsg::ConstSharedPtr & msg);
  void handleCompressedImageCallback(size_t camera_idx, const CompressedImageMsg::ConstSharedPtr & msg);
  void tryPublishSyncedRawImages();
  void tryPublishSyncedCompressedImages();
};

}  // namespace camera_sync

#endif  // camera_sync__MULTI_CAMERA_SYNC_NODE_HPP_
