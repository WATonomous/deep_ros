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

#include "camera_sync/multi_camera_sync_node.hpp"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp_components/register_node_macro.hpp>

#include "deep_msgs/msg/multi_image.hpp"
#include "deep_msgs/msg/multi_image_compressed.hpp"

namespace camera_sync
{

MultiCameraSyncNode::MultiCameraSyncNode(const rclcpp::NodeOptions & options)
#if USE_LIFECYCLE_NODE
: rclcpp_lifecycle::LifecycleNode("multi_camera_sync_node", options)
#else
: Node("multi_camera_sync_node", options)
#endif
, sync_count_(0)
, last_sync_time_(this->get_clock()->now())
, start_time_(std::chrono::steady_clock::now())
{
  RCLCPP_INFO(this->get_logger(), "Initializing Multi-Camera Sync Node");

#if !USE_LIFECYCLE_NODE
  // For regular nodes, initialize immediately
  initializeParameters();
  setupSynchronization();

  RCLCPP_INFO(
    this->get_logger(),
    "Multi-Camera Sync Node initialized with %zu cameras, using %s images",
    camera_topics_.size(),
    use_compressed_ ? "compressed" : "raw");
#else
  // For lifecycle nodes, initialization happens in on_configure
  RCLCPP_INFO(this->get_logger(), "Lifecycle node created, waiting for configuration");
#endif
}

void MultiCameraSyncNode::initializeParameters()
{
  // Declare and get parameters
  this->declare_parameter("camera_topics", std::vector<std::string>{});
  this->declare_parameter("camera_names", std::vector<std::string>{});
  this->declare_parameter("use_compressed", false);
  this->declare_parameter("sync_tolerance_ms", 50.0);
  this->declare_parameter("queue_size", 10);

  camera_topics_ = this->get_parameter("camera_topics").as_string_array();
  camera_names_ = this->get_parameter("camera_names").as_string_array();
  use_compressed_ = this->get_parameter("use_compressed").as_bool();
  sync_tolerance_ms_ = this->get_parameter("sync_tolerance_ms").as_double();
  queue_size_ = this->get_parameter("queue_size").as_int();

  // Validate
  if (camera_topics_.empty() || camera_topics_.size() < 2 || camera_topics_.size() > 12) {
    throw std::runtime_error("Need 2-12 camera topics");
  }

  // Auto-generate names if not provided
  if (camera_names_.empty()) {
    for (size_t i = 0; i < camera_topics_.size(); ++i) {
      camera_names_.push_back("camera_" + std::to_string(i + 1));
    }
  }

  RCLCPP_INFO(
    this->get_logger(), "Configured for %zu %s cameras", camera_topics_.size(), use_compressed_ ? "compressed" : "raw");
}

void MultiCameraSyncNode::setupSynchronization()
{
  const size_t num_cameras = camera_topics_.size();

  if (use_compressed_) {
    setupCompressedSync(num_cameras);
  } else {
    setupRawSync(num_cameras);
  }

  // Setup multi-image publishers
  if (use_compressed_) {
    multi_image_compressed_pub_ =
      this->create_publisher<deep_msgs::msg::MultiImageCompressed>("~/multi_image_compressed", 10);
  } else {
    multi_image_raw_pub_ = this->create_publisher<deep_msgs::msg::MultiImage>("~/multi_image_raw", 10);
  }
}

void MultiCameraSyncNode::setupRawSync(size_t num_cameras)
{
  // Initialize buffers for all cameras
  raw_image_buffers_.resize(num_cameras);

  // Create subscribers with manual synchronization callback
  for (size_t i = 0; i < num_cameras; ++i) {
    auto callback = [this, i](const ImageMsg::ConstSharedPtr & msg) { this->handleRawImageCallback(i, msg); };
    image_subscribers_.push_back(this->create_subscription<ImageMsg>(camera_topics_[i], queue_size_, callback));
  }

  RCLCPP_INFO(this->get_logger(), "Set up raw image synchronization for %zu cameras", num_cameras);
}

void MultiCameraSyncNode::setupCompressedSync(size_t num_cameras)
{
  // Initialize buffers for all cameras
  compressed_image_buffers_.resize(num_cameras);

  // Create subscribers with manual synchronization callback
  for (size_t i = 0; i < num_cameras; ++i) {
    auto callback = [this, i](const CompressedImageMsg::ConstSharedPtr & msg) {
      this->handleCompressedImageCallback(i, msg);
    };
    compressed_subscribers_.push_back(
      this->create_subscription<CompressedImageMsg>(camera_topics_[i], queue_size_, callback));
  }

  RCLCPP_INFO(this->get_logger(), "Set up compressed image synchronization for %zu cameras", num_cameras);
}

// Helper function to create multi-image messages (optimized with move semantics)
template <typename ImageMsgT, typename MultiMsgT>
MultiMsgT createMultiImageMessage(std::vector<typename ImageMsgT::ConstSharedPtr> images)
{
  MultiMsgT msg;
  msg.header.stamp = rclcpp::Clock().now();
  msg.images.reserve(images.size());
  for (auto & img : images) {
    msg.images.push_back(std::move(*img));  // Move instead of copy
  }
  return msg;
}

// Unified processing method that works with timestamps
void MultiCameraSyncNode::processSynchronizedImages(const std::vector<rclcpp::Time> & timestamps)
{
  sync_count_++;
  last_sync_time_ = this->get_clock()->now();

  // Calculate sync statistics
  auto [min_time, max_time] = std::minmax_element(timestamps.begin(), timestamps.end());
  auto time_spread_ms = (*max_time - *min_time).seconds() * 1000.0;

  // Periodic logging
  if (sync_count_ % 100 == 1) {
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    double sync_rate = static_cast<double>(sync_count_) / std::max(elapsed_seconds, 1L);

    RCLCPP_INFO(
      this->get_logger(),
      "Sync #%ld: %zu cameras synchronized, spread: %.1f ms, rate: %.1f Hz",
      sync_count_,
      timestamps.size(),
      time_spread_ms,
      sync_rate);
  }

  // Debug logging
  std::stringstream ss;
  ss << std::fixed << std::setprecision(3);
  for (size_t i = 0; i < timestamps.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << camera_names_[i] << ": " << timestamps[i].seconds();
  }
  ss << " (spread: " << time_spread_ms << " ms)";
  RCLCPP_DEBUG(this->get_logger(), "Synchronized timestamps: %s", ss.str().c_str());
}

#if USE_LIFECYCLE_NODE
// Lifecycle node callbacks
rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn MultiCameraSyncNode::on_configure(
  const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Configuring Multi-Camera Sync Node");

  try {
    initializeParameters();
    setupSynchronization();

    RCLCPP_INFO(
      this->get_logger(),
      "Multi-Camera Sync Node configured with %zu cameras, using %s images",
      camera_topics_.size(),
      use_compressed_ ? "compressed" : "raw");

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to configure: %s", e.what());
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
  }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn MultiCameraSyncNode::on_activate(
  const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Activating Multi-Camera Sync Node");

  // Publishers are automatically activated for lifecycle nodes
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn MultiCameraSyncNode::on_deactivate(
  const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Deactivating Multi-Camera Sync Node");

  // Publishers are automatically deactivated for lifecycle nodes
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn MultiCameraSyncNode::on_cleanup(
  const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Cleaning up Multi-Camera Sync Node");

  // Reset subscribers
  image_subscribers_.clear();
  compressed_subscribers_.clear();

  // Reset publishers
  multi_image_raw_pub_.reset();
  multi_image_compressed_pub_.reset();

  // Clear buffers
  raw_image_buffers_.clear();
  compressed_image_buffers_.clear();

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn MultiCameraSyncNode::on_shutdown(
  const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Shutting down Multi-Camera Sync Node");

  // Cleanup is called automatically before shutdown
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}
#endif

void MultiCameraSyncNode::handleRawImageCallback(size_t camera_idx, const ImageMsg::ConstSharedPtr & msg)
{
  if (camera_idx >= raw_image_buffers_.size()) {
    RCLCPP_WARN(
      this->get_logger(), "Camera index %zu out of bounds (size: %zu)", camera_idx, raw_image_buffers_.size());
    return;
  }

  {
    std::lock_guard<std::mutex> lock(*raw_image_buffers_[camera_idx].mutex);
    uint64_t msg_time_ns = msg->header.stamp.sec * 1000000000ULL + msg->header.stamp.nanosec;
    raw_image_buffers_[camera_idx].buffer[msg_time_ns] = msg;

    RCLCPP_DEBUG(
      this->get_logger(),
      "Camera %zu: received image at time %lu (buffer size: %zu)",
      camera_idx,
      msg_time_ns,
      raw_image_buffers_[camera_idx].buffer.size());

    // Clean old messages outside tolerance window
    auto it = raw_image_buffers_[camera_idx].buffer.begin();
    while (it != raw_image_buffers_[camera_idx].buffer.end()) {
      int64_t time_diff_ns = static_cast<int64_t>(msg_time_ns) - static_cast<int64_t>(it->first);
      if (time_diff_ns > static_cast<int64_t>(sync_tolerance_ms_ * 2 * 1e6)) {
        it = raw_image_buffers_[camera_idx].buffer.erase(it);
      } else {
        ++it;
      }
    }
  }

  tryPublishSyncedRawImages();
}

void MultiCameraSyncNode::handleCompressedImageCallback(
  size_t camera_idx, const CompressedImageMsg::ConstSharedPtr & msg)
{
  if (camera_idx >= compressed_image_buffers_.size()) {
    RCLCPP_WARN(
      this->get_logger(), "Camera index %zu out of bounds (size: %zu)", camera_idx, compressed_image_buffers_.size());
    return;
  }

  {
    std::lock_guard<std::mutex> lock(*compressed_image_buffers_[camera_idx].mutex);
    uint64_t msg_time_ns = msg->header.stamp.sec * 1000000000ULL + msg->header.stamp.nanosec;
    compressed_image_buffers_[camera_idx].buffer[msg_time_ns] = msg;

    RCLCPP_DEBUG(
      this->get_logger(),
      "Camera %zu: received compressed image at time %lu (buffer size: %zu)",
      camera_idx,
      msg_time_ns,
      compressed_image_buffers_[camera_idx].buffer.size());

    // Clean old messages outside tolerance window
    auto it = compressed_image_buffers_[camera_idx].buffer.begin();
    while (it != compressed_image_buffers_[camera_idx].buffer.end()) {
      int64_t time_diff_ns = static_cast<int64_t>(msg_time_ns) - static_cast<int64_t>(it->first);
      if (time_diff_ns > static_cast<int64_t>(sync_tolerance_ms_ * 2 * 1e6)) {
        it = compressed_image_buffers_[camera_idx].buffer.erase(it);
      } else {
        ++it;
      }
    }
  }

  tryPublishSyncedCompressedImages();
}

void MultiCameraSyncNode::tryPublishSyncedRawImages()
{
  std::vector<ImageMsg::ConstSharedPtr> synced_images;
  std::vector<rclcpp::Time> timestamps;
  uint64_t sync_time_ns;

  // Lock all buffers
  for (auto & buffer : raw_image_buffers_) {
    buffer.mutex->lock();
  }

  if (raw_image_buffers_.empty() || raw_image_buffers_[0].buffer.empty()) {
    RCLCPP_DEBUG(this->get_logger(), "No images in buffer yet");
    for (auto & buffer : raw_image_buffers_) {
      buffer.mutex->unlock();
    }
    return;
  }

  // Use the most recent timestamp from camera 0 as reference
  sync_time_ns = raw_image_buffers_[0].buffer.rbegin()->first;

  RCLCPP_DEBUG(this->get_logger(), "Trying to sync with reference time %lu", sync_time_ns);

  // Log buffer sizes
  for (size_t i = 0; i < raw_image_buffers_.size(); ++i) {
    RCLCPP_DEBUG(this->get_logger(), "  Buffer %zu size: %zu", i, raw_image_buffers_[i].buffer.size());
  }

  bool all_found = true;
  for (size_t i = 0; i < raw_image_buffers_.size(); ++i) {
    auto & buffer = raw_image_buffers_[i].buffer;
    int64_t tolerance_ns = static_cast<int64_t>(sync_tolerance_ms_ * 1e6);
    int64_t lower_bound_ns = static_cast<int64_t>(sync_time_ns) - tolerance_ns;
    int64_t upper_bound_ns = static_cast<int64_t>(sync_time_ns) + tolerance_ns;

    // Find the closest timestamp within tolerance window
    auto it = buffer.lower_bound(lower_bound_ns >= 0 ? static_cast<uint64_t>(lower_bound_ns) : 0);

    ImageMsg::ConstSharedPtr best_match = nullptr;
    int64_t best_time_diff = INT64_MAX;

    // Search for the closest match within the tolerance window
    while (it != buffer.end() && static_cast<int64_t>(it->first) <= upper_bound_ns) {
      int64_t time_diff_ns = std::abs(static_cast<int64_t>(sync_time_ns) - static_cast<int64_t>(it->first));
      if (time_diff_ns < best_time_diff) {
        best_time_diff = time_diff_ns;
        best_match = it->second;
      }
      ++it;
    }

    if (best_match != nullptr && best_time_diff <= tolerance_ns) {
      synced_images.push_back(best_match);
      // Find the timestamp key for logging
      for (auto & pair : buffer) {
        if (pair.second == best_match) {
          uint32_t sec = pair.first / 1000000000ULL;
          uint32_t nanosec = pair.first % 1000000000ULL;
          timestamps.push_back(rclcpp::Time(sec, nanosec));
          RCLCPP_DEBUG(
            this->get_logger(),
            "  Camera %zu: found match (time diff: %ld ns)",
            i,
            static_cast<int64_t>(sync_time_ns) - static_cast<int64_t>(pair.first));
          break;
        }
      }
    } else {
      RCLCPP_DEBUG(
        this->get_logger(),
        "  Camera %zu: no match in tolerance window (best diff: %ld ns, tolerance: %ld ns)",
        i,
        best_time_diff,
        tolerance_ns);
      all_found = false;
      break;
    }
  }

  // Unlock all buffers
  for (auto & buffer : raw_image_buffers_) {
    buffer.mutex->unlock();
  }

  if (!all_found || synced_images.size() != raw_image_buffers_.size()) {
    RCLCPP_DEBUG(
      this->get_logger(),
      "Failed to sync: all_found=%d, synced_size=%zu, total_cameras=%zu",
      all_found,
      synced_images.size(),
      raw_image_buffers_.size());
    return;
  }

  // Publish synced images
  RCLCPP_DEBUG(this->get_logger(), "Publishing synced raw images (sync count: %ld)", ++sync_count_);
  processSynchronizedImages(timestamps);
  auto raw_msg = createMultiImageMessage<sensor_msgs::msg::Image, deep_msgs::msg::MultiImage>(synced_images);
  multi_image_raw_pub_->publish(raw_msg);
}

void MultiCameraSyncNode::tryPublishSyncedCompressedImages()
{
  std::vector<CompressedImageMsg::ConstSharedPtr> synced_images;
  std::vector<rclcpp::Time> timestamps;
  uint64_t sync_time_ns;

  // Lock all buffers
  for (auto & buffer : compressed_image_buffers_) {
    buffer.mutex->lock();
  }

  if (compressed_image_buffers_.empty() || compressed_image_buffers_[0].buffer.empty()) {
    RCLCPP_DEBUG(this->get_logger(), "No compressed images in buffer yet");
    for (auto & buffer : compressed_image_buffers_) {
      buffer.mutex->unlock();
    }
    return;
  }

  // Use the most recent timestamp from camera 0 as reference
  sync_time_ns = compressed_image_buffers_[0].buffer.rbegin()->first;

  RCLCPP_DEBUG(this->get_logger(), "Trying to sync compressed with reference time %lu", sync_time_ns);

  // Log buffer sizes
  for (size_t i = 0; i < compressed_image_buffers_.size(); ++i) {
    RCLCPP_DEBUG(
      this->get_logger(), "  Compressed buffer %zu size: %zu", i, compressed_image_buffers_[i].buffer.size());
  }

  bool all_found = true;
  for (size_t i = 0; i < compressed_image_buffers_.size(); ++i) {
    auto & buffer = compressed_image_buffers_[i].buffer;
    int64_t tolerance_ns = static_cast<int64_t>(sync_tolerance_ms_ * 1e6);
    int64_t lower_bound_ns = static_cast<int64_t>(sync_time_ns) - tolerance_ns;
    int64_t upper_bound_ns = static_cast<int64_t>(sync_time_ns) + tolerance_ns;

    // Find the closest timestamp within tolerance window
    auto it = buffer.lower_bound(lower_bound_ns >= 0 ? static_cast<uint64_t>(lower_bound_ns) : 0);

    CompressedImageMsg::ConstSharedPtr best_match = nullptr;
    int64_t best_time_diff = INT64_MAX;

    // Search for the closest match within the tolerance window
    while (it != buffer.end() && static_cast<int64_t>(it->first) <= upper_bound_ns) {
      int64_t time_diff_ns = std::abs(static_cast<int64_t>(sync_time_ns) - static_cast<int64_t>(it->first));
      if (time_diff_ns < best_time_diff) {
        best_time_diff = time_diff_ns;
        best_match = it->second;
      }
      ++it;
    }

    if (best_match != nullptr && best_time_diff <= tolerance_ns) {
      synced_images.push_back(best_match);
      // Find the timestamp key for logging
      for (auto & pair : buffer) {
        if (pair.second == best_match) {
          uint32_t sec = pair.first / 1000000000ULL;
          uint32_t nanosec = pair.first % 1000000000ULL;
          timestamps.push_back(rclcpp::Time(sec, nanosec));
          RCLCPP_DEBUG(
            this->get_logger(),
            "  Camera %zu: found match (time diff: %ld ns)",
            i,
            static_cast<int64_t>(sync_time_ns) - static_cast<int64_t>(pair.first));
          break;
        }
      }
    } else {
      RCLCPP_DEBUG(
        this->get_logger(),
        "  Camera %zu: no match in tolerance window (best diff: %ld ns, tolerance: %ld ns)",
        i,
        best_time_diff,
        tolerance_ns);
      all_found = false;
      break;
    }
  }

  // Unlock all buffers
  for (auto & buffer : compressed_image_buffers_) {
    buffer.mutex->unlock();
  }

  if (!all_found || synced_images.size() != compressed_image_buffers_.size()) {
    RCLCPP_DEBUG(
      this->get_logger(),
      "Failed to sync compressed: all_found=%d, synced_size=%zu, total_cameras=%zu",
      all_found,
      synced_images.size(),
      compressed_image_buffers_.size());
    return;
  }

  // Publish synced images
  RCLCPP_DEBUG(this->get_logger(), "Publishing synced compressed images (sync count: %ld)", ++sync_count_);
  processSynchronizedImages(timestamps);
  auto compressed_msg =
    createMultiImageMessage<sensor_msgs::msg::CompressedImage, deep_msgs::msg::MultiImageCompressed>(synced_images);
  multi_image_compressed_pub_->publish(compressed_msg);
}

}  // namespace camera_sync

// Register the component
RCLCPP_COMPONENTS_REGISTER_NODE(camera_sync::MultiCameraSyncNode)
