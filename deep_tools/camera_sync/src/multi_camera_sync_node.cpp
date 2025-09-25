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
#include <vector>

#include <rclcpp_components/register_node_macro.hpp>

#include "deep_msgs/msg/multi_image.hpp"
#include "deep_msgs/msg/multi_image_raw.hpp"

namespace camera_sync
{

MultiCameraSyncNode::MultiCameraSyncNode(const rclcpp::NodeOptions & options)
: Node("multi_camera_sync_node", options)
, sync_count_(0)
, last_sync_time_(this->get_clock()->now())
, start_time_(std::chrono::steady_clock::now())
{
  RCLCPP_INFO(this->get_logger(), "Initializing Multi-Camera Sync Node");

  initializeParameters();
  setupSynchronization();

  RCLCPP_INFO(
    this->get_logger(),
    "Multi-Camera Sync Node initialized with %zu cameras, using %s images",
    camera_topics_.size(),
    use_compressed_ ? "compressed" : "raw");
}

void MultiCameraSyncNode::initializeParameters()
{
  // Declare and get parameters
  this->declare_parameter("camera_topics", std::vector<std::string>{});
  this->declare_parameter("camera_names", std::vector<std::string>{});
  this->declare_parameter("use_compressed", false);
  this->declare_parameter("sync_tolerance_ms", 50.0);
  this->declare_parameter("queue_size", 10);
  this->declare_parameter("publish_sync_info", true);

  camera_topics_ = this->get_parameter("camera_topics").as_string_array();
  camera_names_ = this->get_parameter("camera_names").as_string_array();
  use_compressed_ = this->get_parameter("use_compressed").as_bool();
  sync_tolerance_ms_ = this->get_parameter("sync_tolerance_ms").as_double();
  queue_size_ = this->get_parameter("queue_size").as_int();
  publish_sync_info_ = this->get_parameter("publish_sync_info").as_bool();

  // Validate
  if (camera_topics_.empty() || camera_topics_.size() < 2 || camera_topics_.size() > 6) {
    throw std::runtime_error("Need 2-6 camera topics");
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
    multi_image_compressed_pub_ = this->create_publisher<deep_msgs::msg::MultiImage>("~/multi_image_compressed", 10);
  } else {
    multi_image_raw_pub_ = this->create_publisher<deep_msgs::msg::MultiImageRaw>("~/multi_image_raw", 10);
  }

  if (publish_sync_info_) {
    sync_info_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/sync_info", 10);
  }
}

void MultiCameraSyncNode::setupRawSync(size_t num_cameras)
{
  // Create subscribers
  image_subscribers_.reserve(num_cameras);
  for (size_t i = 0; i < num_cameras; ++i) {
    image_subscribers_.emplace_back(std::make_unique<ImageSubscriber>(this, camera_topics_[i]));
  }

  // Create synchronizer based on number of cameras
  switch (num_cameras) {
    case 2:
      sync2_raw_ = std::make_unique<message_filters::Synchronizer<ImageSyncPolicy2>>(
        ImageSyncPolicy2(queue_size_), *image_subscribers_[0], *image_subscribers_[1]);
      sync2_raw_->setAgePenalty(sync_tolerance_ms_ / 1000.0);
      sync2_raw_->registerCallback(
        std::bind(&MultiCameraSyncNode::syncCallback2Raw, this, std::placeholders::_1, std::placeholders::_2));
      break;
    case 3:
      sync3_raw_ = std::make_unique<message_filters::Synchronizer<ImageSyncPolicy3>>(
        ImageSyncPolicy3(queue_size_), *image_subscribers_[0], *image_subscribers_[1], *image_subscribers_[2]);
      sync3_raw_->setAgePenalty(sync_tolerance_ms_ / 1000.0);
      sync3_raw_->registerCallback(std::bind(
        &MultiCameraSyncNode::syncCallback3Raw,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3));
      break;
    case 4:
      sync4_raw_ = std::make_unique<message_filters::Synchronizer<ImageSyncPolicy4>>(
        ImageSyncPolicy4(queue_size_),
        *image_subscribers_[0],
        *image_subscribers_[1],
        *image_subscribers_[2],
        *image_subscribers_[3]);
      sync4_raw_->setAgePenalty(sync_tolerance_ms_ / 1000.0);
      sync4_raw_->registerCallback(std::bind(
        &MultiCameraSyncNode::syncCallback4Raw,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4));
      break;
    case 5:
      sync5_raw_ = std::make_unique<message_filters::Synchronizer<ImageSyncPolicy5>>(
        ImageSyncPolicy5(queue_size_),
        *image_subscribers_[0],
        *image_subscribers_[1],
        *image_subscribers_[2],
        *image_subscribers_[3],
        *image_subscribers_[4]);
      sync5_raw_->setAgePenalty(sync_tolerance_ms_ / 1000.0);
      sync5_raw_->registerCallback(std::bind(
        &MultiCameraSyncNode::syncCallback5Raw,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4,
        std::placeholders::_5));
      break;
    case 6:
      sync6_raw_ = std::make_unique<message_filters::Synchronizer<ImageSyncPolicy6>>(
        ImageSyncPolicy6(queue_size_),
        *image_subscribers_[0],
        *image_subscribers_[1],
        *image_subscribers_[2],
        *image_subscribers_[3],
        *image_subscribers_[4],
        *image_subscribers_[5]);
      sync6_raw_->setAgePenalty(sync_tolerance_ms_ / 1000.0);
      sync6_raw_->registerCallback(std::bind(
        &MultiCameraSyncNode::syncCallback6Raw,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4,
        std::placeholders::_5,
        std::placeholders::_6));
      break;
  }
}

void MultiCameraSyncNode::setupCompressedSync(size_t num_cameras)
{
  // Create subscribers
  compressed_subscribers_.reserve(num_cameras);
  for (size_t i = 0; i < num_cameras; ++i) {
    compressed_subscribers_.emplace_back(std::make_unique<CompressedImageSubscriber>(this, camera_topics_[i]));
  }

  // Create synchronizer based on number of cameras
  switch (num_cameras) {
    case 2:
      sync2_compressed_ = std::make_unique<message_filters::Synchronizer<CompressedSyncPolicy2>>(
        CompressedSyncPolicy2(queue_size_), *compressed_subscribers_[0], *compressed_subscribers_[1]);
      sync2_compressed_->setAgePenalty(sync_tolerance_ms_ / 1000.0);
      sync2_compressed_->registerCallback(
        std::bind(&MultiCameraSyncNode::syncCallback2Compressed, this, std::placeholders::_1, std::placeholders::_2));
      break;
    case 3:
      sync3_compressed_ = std::make_unique<message_filters::Synchronizer<CompressedSyncPolicy3>>(
        CompressedSyncPolicy3(queue_size_),
        *compressed_subscribers_[0],
        *compressed_subscribers_[1],
        *compressed_subscribers_[2]);
      sync3_compressed_->setAgePenalty(sync_tolerance_ms_ / 1000.0);
      sync3_compressed_->registerCallback(std::bind(
        &MultiCameraSyncNode::syncCallback3Compressed,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3));
      break;
    case 4:
      sync4_compressed_ = std::make_unique<message_filters::Synchronizer<CompressedSyncPolicy4>>(
        CompressedSyncPolicy4(queue_size_),
        *compressed_subscribers_[0],
        *compressed_subscribers_[1],
        *compressed_subscribers_[2],
        *compressed_subscribers_[3]);
      sync4_compressed_->setAgePenalty(sync_tolerance_ms_ / 1000.0);
      sync4_compressed_->registerCallback(std::bind(
        &MultiCameraSyncNode::syncCallback4Compressed,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4));
      break;
    case 5:
      sync5_compressed_ = std::make_unique<message_filters::Synchronizer<CompressedSyncPolicy5>>(
        CompressedSyncPolicy5(queue_size_),
        *compressed_subscribers_[0],
        *compressed_subscribers_[1],
        *compressed_subscribers_[2],
        *compressed_subscribers_[3],
        *compressed_subscribers_[4]);
      sync5_compressed_->setAgePenalty(sync_tolerance_ms_ / 1000.0);
      sync5_compressed_->registerCallback(std::bind(
        &MultiCameraSyncNode::syncCallback5Compressed,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4,
        std::placeholders::_5));
      break;
    case 6:
      sync6_compressed_ = std::make_unique<message_filters::Synchronizer<CompressedSyncPolicy6>>(
        CompressedSyncPolicy6(queue_size_),
        *compressed_subscribers_[0],
        *compressed_subscribers_[1],
        *compressed_subscribers_[2],
        *compressed_subscribers_[3],
        *compressed_subscribers_[4],
        *compressed_subscribers_[5]);
      sync6_compressed_->setAgePenalty(sync_tolerance_ms_ / 1000.0);
      sync6_compressed_->registerCallback(std::bind(
        &MultiCameraSyncNode::syncCallback6Compressed,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4,
        std::placeholders::_5,
        std::placeholders::_6));
      break;
  }
}

// Helper function to create multi-image messages
template <typename ImageMsgT, typename MultiMsgT>
MultiMsgT createMultiImageMessage(const std::vector<typename ImageMsgT::ConstSharedPtr> & images)
{
  MultiMsgT msg;
  msg.header.stamp = rclcpp::Clock().now();
  msg.images.reserve(images.size());
  for (const auto & img : images) {
    msg.images.push_back(*img);  // Deep copy onto new synced message
  }
  return msg;
}

// Raw image callbacks
void MultiCameraSyncNode::syncCallback2Raw(const ImageMsg::ConstSharedPtr & img1, const ImageMsg::ConstSharedPtr & img2)
{
  std::vector<rclcpp::Time> timestamps = {rclcpp::Time(img1->header.stamp), rclcpp::Time(img2->header.stamp)};
  processSynchronizedImages(timestamps);

  // Create and publish multi-image message
  std::vector<ImageMsg::ConstSharedPtr> images = {img1, img2};
  auto raw_msg = createMultiImageMessage<sensor_msgs::msg::Image, deep_msgs::msg::MultiImageRaw>(images);
  multi_image_raw_pub_->publish(raw_msg);
}

void MultiCameraSyncNode::syncCallback3Raw(
  const ImageMsg::ConstSharedPtr & img1, const ImageMsg::ConstSharedPtr & img2, const ImageMsg::ConstSharedPtr & img3)
{
  std::vector<rclcpp::Time> timestamps = {
    rclcpp::Time(img1->header.stamp), rclcpp::Time(img2->header.stamp), rclcpp::Time(img3->header.stamp)};
  processSynchronizedImages(timestamps);

  // Create and publish multi-image message
  std::vector<ImageMsg::ConstSharedPtr> images = {img1, img2, img3};
  auto raw_msg = createMultiImageMessage<sensor_msgs::msg::Image, deep_msgs::msg::MultiImageRaw>(images);
  multi_image_raw_pub_->publish(raw_msg);
}

void MultiCameraSyncNode::syncCallback4Raw(
  const ImageMsg::ConstSharedPtr & img1,
  const ImageMsg::ConstSharedPtr & img2,
  const ImageMsg::ConstSharedPtr & img3,
  const ImageMsg::ConstSharedPtr & img4)
{
  std::vector<rclcpp::Time> timestamps = {
    rclcpp::Time(img1->header.stamp),
    rclcpp::Time(img2->header.stamp),
    rclcpp::Time(img3->header.stamp),
    rclcpp::Time(img4->header.stamp)};
  processSynchronizedImages(timestamps);

  // Create and publish multi-image message
  std::vector<ImageMsg::ConstSharedPtr> images = {img1, img2, img3, img4};
  auto raw_msg = createMultiImageMessage<sensor_msgs::msg::Image, deep_msgs::msg::MultiImageRaw>(images);
  multi_image_raw_pub_->publish(raw_msg);
}

void MultiCameraSyncNode::syncCallback5Raw(
  const ImageMsg::ConstSharedPtr & img1,
  const ImageMsg::ConstSharedPtr & img2,
  const ImageMsg::ConstSharedPtr & img3,
  const ImageMsg::ConstSharedPtr & img4,
  const ImageMsg::ConstSharedPtr & img5)
{
  std::vector<rclcpp::Time> timestamps = {
    rclcpp::Time(img1->header.stamp),
    rclcpp::Time(img2->header.stamp),
    rclcpp::Time(img3->header.stamp),
    rclcpp::Time(img4->header.stamp),
    rclcpp::Time(img5->header.stamp)};
  processSynchronizedImages(timestamps);

  // Create and publish multi-image message
  std::vector<ImageMsg::ConstSharedPtr> images = {img1, img2, img3, img4, img5};
  auto raw_msg = createMultiImageMessage<sensor_msgs::msg::Image, deep_msgs::msg::MultiImageRaw>(images);
  multi_image_raw_pub_->publish(raw_msg);
}

void MultiCameraSyncNode::syncCallback6Raw(
  const ImageMsg::ConstSharedPtr & img1,
  const ImageMsg::ConstSharedPtr & img2,
  const ImageMsg::ConstSharedPtr & img3,
  const ImageMsg::ConstSharedPtr & img4,
  const ImageMsg::ConstSharedPtr & img5,
  const ImageMsg::ConstSharedPtr & img6)
{
  std::vector<rclcpp::Time> timestamps = {
    rclcpp::Time(img1->header.stamp),
    rclcpp::Time(img2->header.stamp),
    rclcpp::Time(img3->header.stamp),
    rclcpp::Time(img4->header.stamp),
    rclcpp::Time(img5->header.stamp),
    rclcpp::Time(img6->header.stamp)};
  processSynchronizedImages(timestamps);

  // Create and publish multi-image message
  std::vector<ImageMsg::ConstSharedPtr> images = {img1, img2, img3, img4, img5, img6};
  auto raw_msg = createMultiImageMessage<sensor_msgs::msg::Image, deep_msgs::msg::MultiImageRaw>(images);
  multi_image_raw_pub_->publish(raw_msg);
}

// Compressed image callbacks
void MultiCameraSyncNode::syncCallback2Compressed(
  const CompressedImageMsg::ConstSharedPtr & img1, const CompressedImageMsg::ConstSharedPtr & img2)
{
  std::vector<rclcpp::Time> timestamps = {rclcpp::Time(img1->header.stamp), rclcpp::Time(img2->header.stamp)};
  processSynchronizedImages(timestamps);

  // Create and publish multi-image message
  std::vector<CompressedImageMsg::ConstSharedPtr> images = {img1, img2};
  auto compressed_msg = createMultiImageMessage<sensor_msgs::msg::CompressedImage, deep_msgs::msg::MultiImage>(images);
  multi_image_compressed_pub_->publish(compressed_msg);
}

void MultiCameraSyncNode::syncCallback3Compressed(
  const CompressedImageMsg::ConstSharedPtr & img1,
  const CompressedImageMsg::ConstSharedPtr & img2,
  const CompressedImageMsg::ConstSharedPtr & img3)
{
  std::vector<rclcpp::Time> timestamps = {
    rclcpp::Time(img1->header.stamp), rclcpp::Time(img2->header.stamp), rclcpp::Time(img3->header.stamp)};
  processSynchronizedImages(timestamps);

  // Create and publish multi-image message
  std::vector<CompressedImageMsg::ConstSharedPtr> images = {img1, img2, img3};
  auto compressed_msg = createMultiImageMessage<sensor_msgs::msg::CompressedImage, deep_msgs::msg::MultiImage>(images);
  multi_image_compressed_pub_->publish(compressed_msg);
}

void MultiCameraSyncNode::syncCallback4Compressed(
  const CompressedImageMsg::ConstSharedPtr & img1,
  const CompressedImageMsg::ConstSharedPtr & img2,
  const CompressedImageMsg::ConstSharedPtr & img3,
  const CompressedImageMsg::ConstSharedPtr & img4)
{
  std::vector<rclcpp::Time> timestamps = {
    rclcpp::Time(img1->header.stamp),
    rclcpp::Time(img2->header.stamp),
    rclcpp::Time(img3->header.stamp),
    rclcpp::Time(img4->header.stamp)};
  processSynchronizedImages(timestamps);

  // Create and publish multi-image message
  std::vector<CompressedImageMsg::ConstSharedPtr> images = {img1, img2, img3, img4};
  auto compressed_msg = createMultiImageMessage<sensor_msgs::msg::CompressedImage, deep_msgs::msg::MultiImage>(images);
  multi_image_compressed_pub_->publish(compressed_msg);
}

void MultiCameraSyncNode::syncCallback5Compressed(
  const CompressedImageMsg::ConstSharedPtr & img1,
  const CompressedImageMsg::ConstSharedPtr & img2,
  const CompressedImageMsg::ConstSharedPtr & img3,
  const CompressedImageMsg::ConstSharedPtr & img4,
  const CompressedImageMsg::ConstSharedPtr & img5)
{
  std::vector<rclcpp::Time> timestamps = {
    rclcpp::Time(img1->header.stamp),
    rclcpp::Time(img2->header.stamp),
    rclcpp::Time(img3->header.stamp),
    rclcpp::Time(img4->header.stamp),
    rclcpp::Time(img5->header.stamp)};
  processSynchronizedImages(timestamps);

  // Create and publish multi-image message
  std::vector<CompressedImageMsg::ConstSharedPtr> images = {img1, img2, img3, img4, img5};
  auto compressed_msg = createMultiImageMessage<sensor_msgs::msg::CompressedImage, deep_msgs::msg::MultiImage>(images);
  multi_image_compressed_pub_->publish(compressed_msg);
}

void MultiCameraSyncNode::syncCallback6Compressed(
  const CompressedImageMsg::ConstSharedPtr & img1,
  const CompressedImageMsg::ConstSharedPtr & img2,
  const CompressedImageMsg::ConstSharedPtr & img3,
  const CompressedImageMsg::ConstSharedPtr & img4,
  const CompressedImageMsg::ConstSharedPtr & img5,
  const CompressedImageMsg::ConstSharedPtr & img6)
{
  std::vector<rclcpp::Time> timestamps = {
    rclcpp::Time(img1->header.stamp),
    rclcpp::Time(img2->header.stamp),
    rclcpp::Time(img3->header.stamp),
    rclcpp::Time(img4->header.stamp),
    rclcpp::Time(img5->header.stamp),
    rclcpp::Time(img6->header.stamp)};
  processSynchronizedImages(timestamps);

  // Create and publish multi-image message
  std::vector<CompressedImageMsg::ConstSharedPtr> images = {img1, img2, img3, img4, img5, img6};
  auto compressed_msg = createMultiImageMessage<sensor_msgs::msg::CompressedImage, deep_msgs::msg::MultiImage>(images);
  multi_image_compressed_pub_->publish(compressed_msg);
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

}  // namespace camera_sync

// Register the component
RCLCPP_COMPONENTS_REGISTER_NODE(camera_sync::MultiCameraSyncNode)

// Main function
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<camera_sync::MultiCameraSyncNode>();

  RCLCPP_INFO(node->get_logger(), "Multi-Camera Sync Node started");

  try {
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node->get_logger(), "Node crashed: %s", e.what());
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
