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

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <std_msgs/msg/header.hpp>

namespace deep_yolo_inference
{

enum class Provider
{
  TENSORRT,
  CUDA,
  CPU
};

struct ImageMeta
{
  int original_width = 0;
  int original_height = 0;
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
};

struct QueuedImage
{
  cv::Mat bgr;
  std_msgs::msg::Header header;
};

struct PackedInput
{
  std::vector<float> data;
  std::vector<size_t> shape;
};

struct YoloParams
{
  std::string model_path;
  std::string input_image_topic{"/camera/image_raw"};
  std::vector<std::string> camera_topics;
  std::string input_transport{"raw"};
  std::string input_qos_reliability{"best_effort"};
  std::string output_detections_topic{"/detections"};
  std::string class_names_path;
  int input_width{640};
  int input_height{640};
  bool use_letterbox{false};
  int batch_size_limit{3};
  int queue_size{10};
  double score_threshold{0.25};
  double nms_iou_threshold{0.45};
  std::string preboxed_format{"cxcywh"};
  std::string preferred_provider{"tensorrt"};
  int device_id{0};
  bool warmup_tensor_shapes{true};
  bool enable_trt_engine_cache{false};
  std::string trt_engine_cache_path{"/tmp/deep_ros_ort_trt_cache"};
};

}  // namespace deep_yolo_inference
