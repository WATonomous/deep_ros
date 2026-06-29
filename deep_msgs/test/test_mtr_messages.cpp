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

#include <gtest/gtest.h>

#include <deep_msgs/msg/map_polyline.hpp>
#include <deep_msgs/msg/mtr_prediction_array.hpp>
#include <deep_msgs/msg/mtr_scene.hpp>

TEST(MtrMessages, PreserveCorrelationAndSemanticMapData)
{
  deep_msgs::msg::MtrScene scene;
  scene.request_id = "request-7";
  scene.has_ego_pose = true;
  scene.has_map = true;

  deep_msgs::msg::MapPolyline centerline;
  centerline.lanelet_id = 42;
  centerline.semantic_type = deep_msgs::msg::MapPolyline::CENTERLINE;
  centerline.points.resize(2);
  scene.map_polylines.push_back(centerline);

  EXPECT_EQ(scene.request_id, "request-7");
  ASSERT_EQ(scene.map_polylines.size(), 1u);
  EXPECT_EQ(scene.map_polylines.front().lanelet_id, 42);
  EXPECT_EQ(scene.map_polylines.front().semantic_type, deep_msgs::msg::MapPolyline::CENTERLINE);

  deep_msgs::msg::MtrPredictionArray result;
  result.request_id = scene.request_id;
  EXPECT_EQ(result.request_id, scene.request_id);
}
