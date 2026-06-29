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

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <catch2/catch_test_macros.hpp>
#include <deep_msgs/msg/mtr_prediction_array.hpp>
#include <deep_msgs/msg/mtr_scene.hpp>
#include <deep_mtr/deep_mtr_node.hpp>
#include <deep_test/deep_test.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>

TEST_CASE("DeepMtrNode runs as an empty lifecycle skeleton", "[deep_mtr][lifecycle]")
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<deep_mtr::DeepMtrNode>();

  REQUIRE(node->configure().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  REQUIRE(node->activate().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  REQUIRE(node->deactivate().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  REQUIRE(node->cleanup().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}

TEST_CASE("DeepMtrNode never fabricates a prediction", "[deep_mtr][placeholder]")
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<deep_mtr::DeepMtrNode>();
  auto client = std::make_shared<rclcpp::Node>("deep_mtr_test_client");
  std::atomic<int> result_count{0};
  auto result_sub = client->create_subscription<deep_msgs::msg::MtrPredictionArray>(
    "/mtr/predictions", 10, [&result_count](deep_msgs::msg::MtrPredictionArray::ConstSharedPtr) { ++result_count; });
  auto scene_pub = client->create_publisher<deep_msgs::msg::MtrScene>("/mtr/scenes", 10);

  REQUIRE(node->configure().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  REQUIRE(node->activate().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());
  executor.add_node(client);
  deep_msgs::msg::MtrScene scene;
  scene.request_id = "must-not-produce-output";
  scene_pub->publish(scene);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  REQUIRE(result_count.load() == 0);
  node->deactivate();
  node->cleanup();
}
