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

namespace deep_ros::test
{

ROS2Initializer::ROS2Initializer()
{
  rclcpp::init(0, nullptr);
}

ROS2Initializer::~ROS2Initializer()
{
  rclcpp::shutdown();
}

TestExecutorFixture::TestExecutorFixture()
: ROS2Initializer()
{}

TestExecutorFixture::~TestExecutorFixture()
{
  // Cancel executor and wait for thread to finish
  executor_.cancel();

  if (spin_thread_.joinable()) {
    spin_thread_.join();
  }

  // Clear any remaining nodes from executor before shutdown
  executor_.remove_all_nodes();
}

void TestExecutorFixture::start_spinning()
{
  if (!spin_thread_.joinable()) {
    spin_thread_ = std::thread([this]() { executor_.spin(); });
  }
}

}  // namespace deep_ros::test
