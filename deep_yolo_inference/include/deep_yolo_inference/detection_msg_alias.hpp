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

#include <std_msgs/msg/header.hpp>

// Prefer deep_msgs if the detection messages are present. Fallback to vision_msgs
// so the node can still build in this workspace.
#if __has_include(<deep_msgs/msg/detection2_d_array.hpp>)
  #include <deep_msgs/msg/detection2_d.hpp>
  #include <deep_msgs/msg/detection2_d_array.hpp>

namespace deep_yolo_inference
{
using Detection2DMsg = deep_msgs::msg::Detection2D;
using Detection2DArrayMsg = deep_msgs::msg::Detection2DArray;
}  // namespace deep_yolo_inference
#else
  #include <vision_msgs/msg/detection2_d.hpp>
  #include <vision_msgs/msg/detection2_d_array.hpp>
  #include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

namespace deep_yolo_inference
{
using Detection2DMsg = vision_msgs::msg::Detection2D;
using Detection2DArrayMsg = vision_msgs::msg::Detection2DArray;
}  // namespace deep_yolo_inference
#endif

namespace deep_yolo_inference
{

struct SimpleDetection
{
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float score = 0.0f;
  int32_t class_id = -1;
};

}  // namespace deep_yolo_inference
