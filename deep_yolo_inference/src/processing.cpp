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

#include "deep_yolo_inference/processing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace deep_yolo_inference
{

ImagePreprocessor::ImagePreprocessor(const YoloParams & params)
: params_(params)
{}

cv::Mat ImagePreprocessor::preprocess(const cv::Mat & bgr, ImageMeta & meta) const
{
  if (bgr.empty()) {
    throw std::runtime_error("Input image is empty");
  }

  meta.original_width = bgr.cols;
  meta.original_height = bgr.rows;

  cv::Mat resized;
  if (params_.use_letterbox) {
    const float scale = std::min(
      static_cast<float>(params_.input_width) / static_cast<float>(bgr.cols),
      static_cast<float>(params_.input_height) / static_cast<float>(bgr.rows));
    const int new_w = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
    const int new_h = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));
    cv::resize(bgr, resized, cv::Size(new_w, new_h));

    const int pad_w = params_.input_width - new_w;
    const int pad_h = params_.input_height - new_h;
    const int pad_left = pad_w / 2;
    const int pad_right = pad_w - pad_left;
    const int pad_top = pad_h / 2;
    const int pad_bottom = pad_h - pad_top;

    cv::copyMakeBorder(
      resized, resized, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    meta.scale_x = meta.scale_y = scale;
    meta.pad_x = static_cast<float>(pad_left);
    meta.pad_y = static_cast<float>(pad_top);
  } else {
    cv::resize(bgr, resized, cv::Size(params_.input_width, params_.input_height));
    meta.scale_x = static_cast<float>(meta.original_width) / static_cast<float>(params_.input_width);
    meta.scale_y = static_cast<float>(meta.original_height) / static_cast<float>(params_.input_height);
    meta.pad_x = meta.pad_y = 0.0f;
  }

  cv::Mat float_image;
  resized.convertTo(float_image, CV_32F, 1.0 / 255.0);
  cv::cvtColor(float_image, float_image, cv::COLOR_BGR2RGB);
  return float_image;
}

const PackedInput & ImagePreprocessor::pack(const std::vector<cv::Mat> & images) const
{
  auto & packed = packed_input_cache_;
  packed.data.clear();
  packed.shape.clear();
  if (images.empty()) {
    return packed;
  }

  const size_t batch = images.size();
  const size_t channels = 3;
  const size_t height = images[0].rows;
  const size_t width = images[0].cols;
  const size_t image_size = channels * height * width;
  const size_t required = batch * image_size;

  packed.shape = {batch, channels, height, width};
  packed.data.resize(required);

  std::array<cv::Mat, 3> channel_planes;
  const size_t plane_elements = height * width;
  const size_t plane_bytes = plane_elements * sizeof(float);
  for (auto & plane : channel_planes) {
    plane.create(static_cast<int>(height), static_cast<int>(width), CV_32F);
  }

  for (size_t b = 0; b < batch; ++b) {
    const cv::Mat & img = images[b];
    if (img.channels() != 3 || img.type() != CV_32FC3) {
      throw std::runtime_error("Preprocessed image must be CV_32FC3");
    }

    cv::split(img, channel_planes.data());
    float * batch_base = packed.data.data() + b * image_size;
    for (size_t c = 0; c < channels; ++c) {
      const float * src_ptr = channel_planes[c].ptr<float>();
      float * dst_ptr = batch_base + c * plane_elements;
      std::memcpy(dst_ptr, src_ptr, plane_bytes);
    }
  }

  return packed;
}

Postprocessor::Postprocessor(const YoloParams & params, const std::vector<std::string> & class_names)
: params_(params)
, class_names_(class_names)
{}

std::vector<std::vector<SimpleDetection>> Postprocessor::decode(
  const deep_ros::Tensor & output, const std::vector<ImageMeta> & metas) const
{
  std::vector<std::vector<SimpleDetection>> batch_detections;

  const auto & shape = output.shape();
  if (shape.size() != 3) {
    throw std::runtime_error("Unexpected output shape rank; expected 3D output tensor");
  }

  const size_t batch = shape[0];
  const size_t dim1 = shape[1];
  const size_t dim2 = shape[2];

  const float * data = output.data_as<float>();
  if (!data) {
    throw std::runtime_error("Output tensor has null data");
  }

  const size_t class_count = class_names_.size();

  auto decode_preboxed = [&](size_t num_detections, size_t values, bool values_first) {
    if (values < 6) {
      throw std::runtime_error("Output last dimension must be at least 6");
    }
    batch_detections.clear();
    batch_detections.reserve(std::min(batch, metas.size()));
    for (size_t b = 0; b < batch && b < metas.size(); ++b) {
      std::vector<SimpleDetection> dets;
      dets.reserve(num_detections);
      for (size_t i = 0; i < num_detections; ++i) {
        const auto read_val = [&](size_t v_idx) {
          if (values_first) {
            return data[(b * values + v_idx) * num_detections + i];
          }
          return data[(b * num_detections + i) * values + v_idx];
        };

        const float obj = read_val(4);
        if (obj < static_cast<float>(params_.score_threshold)) {
          continue;
        }

        SimpleDetection det;
        if (params_.preboxed_format == "xyxy") {
          const float x1 = read_val(0);
          const float y1 = read_val(1);
          const float x2 = read_val(2);
          const float y2 = read_val(3);
          det.x = (x1 + x2) * 0.5f;
          det.y = (y1 + y2) * 0.5f;
          det.width = x2 - x1;
          det.height = y2 - y1;
        } else {
          det.x = read_val(0);
          det.y = read_val(1);
          det.width = read_val(2);
          det.height = read_val(3);
        }
        det.score = obj;
        det.class_id = static_cast<int32_t>(std::round(read_val(5)));

        adjustToOriginal(det, metas[b]);
        dets.push_back(det);
      }
      batch_detections.push_back(applyNms(dets));
    }
  };

  if (dim2 <= 8) {
    decode_preboxed(dim1, dim2, false);
    return batch_detections;
  }

  if (dim1 <= 8) {
    decode_preboxed(dim2, dim1, true);
    return batch_detections;
  }

  const bool channels_first = dim1 < dim2;
  const size_t channels = channels_first ? dim1 : dim2;
  const size_t anchors = channels_first ? dim2 : dim1;

  if (channels < 5) {
    throw std::runtime_error("Unsupported YOLO output layout; channels dimension too small");
  }

  const bool has_objectness = class_count > 0 ? (channels > class_count + 4) : (channels >= 6);
  const size_t cls_start = has_objectness ? 5 : 4;
  const size_t available_cls = channels > cls_start ? channels - cls_start : 0;
  const size_t num_classes = class_count > 0 ? std::min(class_count, available_cls) : available_cls;

  batch_detections.reserve(std::min(batch, metas.size()));
  for (size_t b = 0; b < batch && b < metas.size(); ++b) {
    std::vector<SimpleDetection> dets;
    dets.reserve(anchors / 4);
    const size_t batch_offset = b * channels * anchors;

    for (size_t a = 0; a < anchors; ++a) {
      auto read = [&](size_t c) -> float {
        if (channels_first) {
          return data[batch_offset + c * anchors + a];
        }
        return data[batch_offset + a * channels + c];
      };

      const float cx = read(0);
      const float cy = read(1);
      const float w = read(2);
      const float h = read(3);
      const float obj = has_objectness ? read(4) : 1.0f;

      float best_cls_score = num_classes > 0 ? 0.0f : 1.0f;
      size_t best_cls = 0;
      for (size_t cls = 0; cls < num_classes; ++cls) {
        const float cls_score = read(cls_start + cls);
        if (cls_score > best_cls_score) {
          best_cls_score = cls_score;
          best_cls = cls;
        }
      }

      const float score = obj * best_cls_score;
      if (score < static_cast<float>(params_.score_threshold)) {
        continue;
      }

      SimpleDetection det;
      det.x = cx;
      det.y = cy;
      det.width = w;
      det.height = h;
      det.score = score;
      det.class_id = static_cast<int32_t>(best_cls);

      adjustToOriginal(det, metas[b]);
      dets.push_back(det);
    }

    batch_detections.push_back(applyNms(dets));
  }

  return batch_detections;
}

void Postprocessor::adjustToOriginal(SimpleDetection & det, const ImageMeta & meta) const
{
  float cx = det.x;
  float cy = det.y;
  float w = det.width;
  float h = det.height;

  cx -= meta.pad_x;
  cy -= meta.pad_y;

  if (params_.use_letterbox) {
    const float inv_scale = meta.scale_x > 0.0f ? 1.0f / meta.scale_x : 1.0f;
    cx *= inv_scale;
    cy *= inv_scale;
    w *= inv_scale;
    h *= inv_scale;
  } else {
    cx *= meta.scale_x;
    cy *= meta.scale_y;
    w *= meta.scale_x;
    h *= meta.scale_y;
  }

  det.x = std::max(0.0f, cx - w * 0.5f);
  det.y = std::max(0.0f, cy - h * 0.5f);
  det.width = w;
  det.height = h;

  det.x = std::min(det.x, static_cast<float>(meta.original_width));
  det.y = std::min(det.y, static_cast<float>(meta.original_height));
  det.width = std::min(det.width, static_cast<float>(meta.original_width) - det.x);
  det.height = std::min(det.height, static_cast<float>(meta.original_height) - det.y);
}

float Postprocessor::iou(const SimpleDetection & a, const SimpleDetection & b)
{
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);

  const float inter_w = std::max(0.0f, x2 - x1);
  const float inter_h = std::max(0.0f, y2 - y1);
  const float inter_area = inter_w * inter_h;
  const float area_a = a.width * a.height;
  const float area_b = b.width * b.height;
  const float union_area = area_a + area_b - inter_area;

  if (union_area <= 0.0f) {
    return 0.0f;
  }
  return inter_area / union_area;
}

std::vector<SimpleDetection> Postprocessor::applyNms(std::vector<SimpleDetection> dets) const
{
  std::vector<SimpleDetection> result;
  if (dets.empty()) {
    return result;
  }

  std::stable_sort(
    dets.begin(), dets.end(), [](const SimpleDetection & a, const SimpleDetection & b) { return a.score > b.score; });

  std::vector<bool> suppressed(dets.size(), false);
  for (size_t i = 0; i < dets.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    result.push_back(dets[i]);
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }
      if (dets[i].class_id != dets[j].class_id) {
        continue;
      }
      if (iou(dets[i], dets[j]) > static_cast<float>(params_.nms_iou_threshold)) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

void Postprocessor::fillDetectionMessage(
  const std_msgs::msg::Header & header,
  const std::vector<SimpleDetection> & detections,
  const ImageMeta & /*meta*/,
  Detection2DArrayMsg & out_msg) const
{
  out_msg.header = header;

#if __has_include(<deep_msgs/msg/detection2_d_array.hpp>)
  out_msg.detections.clear();
  out_msg.detections.reserve(detections.size());
  for (const auto & det : detections) {
    Detection2DMsg d;
    d.x = det.x;
    d.y = det.y;
    d.width = det.width;
    d.height = det.height;
    d.score = det.score;
    d.class_id = det.class_id;
    d.label = classLabel(det.class_id);
    out_msg.detections.push_back(d);
  }
#else
  out_msg.detections.clear();
  out_msg.detections.reserve(detections.size());
  for (const auto & det : detections) {
    Detection2DMsg d;
    d.header = header;
    d.bbox.center.position.x = det.x + det.width * 0.5;
    d.bbox.center.position.y = det.y + det.height * 0.5;
    d.bbox.size_x = det.width;
    d.bbox.size_y = det.height;

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = classLabel(det.class_id);
    hyp.hypothesis.score = det.score;
    d.results.push_back(hyp);
    out_msg.detections.push_back(d);
  }
#endif
}

std::string Postprocessor::classLabel(int class_id) const
{
  if (class_id >= 0 && static_cast<size_t>(class_id) < class_names_.size()) {
    return class_names_[static_cast<size_t>(class_id)];
  }
  return std::to_string(class_id);
}

}  // namespace deep_yolo_inference
