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

#include "test_fixtures/test_executor_fixture.hpp"

#include <atomic>
#include <mutex>

#include <lifecycle_msgs/msg/state.hpp>

namespace deep_ros::test
{

// Static members for safe one-time initialization
static std::mutex ros_init_mutex;
static std::atomic<int> ros_init_count{0};
static std::atomic<bool> ros_initialized{false};

ROS2Initializer::ROS2Initializer()
{
  std::lock_guard<std::mutex> lock(ros_init_mutex);
  if (!ros_initialized.load()) {
    rclcpp::init(0, nullptr);
    ros_initialized.store(true);
  }
  ros_init_count++;
}

ROS2Initializer::~ROS2Initializer()
{
  std::lock_guard<std::mutex> lock(ros_init_mutex);
  ros_init_count--;
  // Only shutdown when the last instance is destroyed
  if (ros_init_count.load() == 0 && ros_initialized.load()) {
    rclcpp::shutdown();
    ros_initialized.store(false);
  }
}

TestExecutorFixture::TestExecutorFixture()
: ROS2Initializer()
{}

TestExecutorFixture::~TestExecutorFixture()
{
  stop_spinning();

  // Now that spinning has stopped, we can safely deactivate lifecycle nodes
  for (auto & node : lifecycle_nodes_) {
    if (!node) {
      continue;
    }

    auto state = node->get_current_state().id();
    if (state == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
      node->deactivate();
    }
  }

  lifecycle_nodes_.clear();
}

void TestExecutorFixture::start_spinning()
{
  if (!spin_thread_.joinable()) {
    should_spin_ = true;
    spin_thread_ = std::thread([this]() {
      // Use spin_once with timeout instead of blocking spin()
      // This allows the thread to exit when should_spin_ is set to false
      while (should_spin_.load()) {
        executor_.spin_once(std::chrono::milliseconds(100));
      }
    });
  }
}

void TestExecutorFixture::stop_spinning()
{
  should_spin_ = false;
  executor_.cancel();

  if (spin_thread_.joinable()) {
    // Give it a moment to exit the loop
    if (!spin_thread_.joinable()) {
      return;  // Already joined
    }
    spin_thread_.join();
  }
}

}  // namespace deep_ros::test
