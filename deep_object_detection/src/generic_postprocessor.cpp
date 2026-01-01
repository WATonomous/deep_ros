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

#include "deep_object_detection/generic_postprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace deep_object_detection
{

GenericPostprocessor::GenericPostprocessor(
  const PostprocessingConfig & config,
  const OutputLayout & layout,
  BboxFormat bbox_format,
  int num_classes,
  const std::vector<std::string> & class_names,
  bool use_letterbox)
: config_(config)
, layout_(layout)
, bbox_format_(bbox_format)
, num_classes_(num_classes)
, class_names_(class_names)
, use_letterbox_(use_letterbox)
{}

GenericPostprocessor::OutputLayout GenericPostprocessor::detectLayout(const std::vector<size_t> & output_shape)
{
  OutputLayout layout;
  layout.shape = output_shape;
  layout.auto_detect = true;

  if (output_shape.size() < 2) {
    throw std::runtime_error("Output tensor must have at least 2 dimensions");
  }

  // Default: assume [batch, detections, features] or [batch, features, detections]
  layout.batch_dim = 0;
  
  // Determine layout based on shape
  if (output_shape.size() == 2) {
    // [batch, features] - single detection per batch
    layout.detection_dim = 0;  // Batch is also detection
    layout.feature_dim = 1;
    layout.bbox_start_idx = 0;
    layout.bbox_count = 4;
    layout.score_idx = 4;
    layout.class_idx = 5;
  } else if (output_shape.size() == 3) {
    // [batch, dim1, dim2] - need to determine which is detections vs features
    size_t dim1 = output_shape[1];
    size_t dim2 = output_shape[2];
    
    if (dim1 <= 8 && dim2 > 8) {
      // [batch, features, detections] - features first
      layout.detection_dim = 2;
      layout.feature_dim = 1;
    } else if (dim1 > 8 && dim2 <= 8) {
      // [batch, detections, features] - detections first
      layout.detection_dim = 1;
      layout.feature_dim = 2;
    } else {
      // Ambiguous - default to [batch, detections, features]
      layout.detection_dim = 1;
      layout.feature_dim = 2;
    }
    
    layout.bbox_start_idx = 0;
    layout.bbox_count = 4;
    layout.score_idx = 4;
    layout.class_idx = 5;
  } else {
    // Higher dimensional - use defaults
    layout.detection_dim = 1;
    layout.feature_dim = 2;
    layout.bbox_start_idx = 0;
    layout.bbox_count = 4;
    layout.score_idx = 4;
    layout.class_idx = 5;
  }

  return layout;
}

float GenericPostprocessor::extractValue(
  const float * data,
  size_t batch_idx,
  size_t detection_idx,
  size_t feature_idx,
  const std::vector<size_t> & shape) const
{
  if (shape.size() == 2) {
    // [batch, features]
    return data[batch_idx * shape[1] + feature_idx];
  } else if (shape.size() == 3) {
    if (layout_.detection_dim == 1 && layout_.feature_dim == 2) {
      // [batch, detections, features]
      return data[(batch_idx * shape[1] + detection_idx) * shape[2] + feature_idx];
    } else if (layout_.detection_dim == 2 && layout_.feature_dim == 1) {
      // [batch, features, detections]
      return data[(batch_idx * shape[1] + feature_idx) * shape[2] + detection_idx];
    }
  }
  
  // Generic indexing for higher dimensions
  size_t index = 0;
  size_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    size_t dim_idx = 0;
    if (i == static_cast<int>(layout_.batch_dim)) {
      dim_idx = batch_idx;
    } else if (i == static_cast<int>(layout_.detection_dim)) {
      dim_idx = detection_idx;
    } else if (i == static_cast<int>(layout_.feature_dim)) {
      dim_idx = feature_idx;
    }
    index += dim_idx * stride;
    stride *= shape[i];
  }
  
  return data[index];
}

void GenericPostprocessor::convertBbox(
  const float * bbox_data,
  size_t batch_idx,
  size_t detection_idx,
  const std::vector<size_t> & shape,
  SimpleDetection & det) const
{
  // Extract bbox coordinates
  float coords[4];
  for (size_t i = 0; i < 4; ++i) {
    coords[i] = extractValue(
      bbox_data, batch_idx, detection_idx, layout_.bbox_start_idx + i, shape);
  }

  // Convert based on format
  switch (bbox_format_) {
    case BboxFormat::XYXY: {
      // [x1, y1, x2, y2] -> center + size
      det.x = (coords[0] + coords[2]) * 0.5f;
      det.y = (coords[1] + coords[3]) * 0.5f;
      det.width = coords[2] - coords[0];
      det.height = coords[3] - coords[1];
      break;
    }
    case BboxFormat::XYWH: {
      // [x, y, w, h] -> center + size
      det.x = coords[0] + coords[2] * 0.5f;
      det.y = coords[1] + coords[3] * 0.5f;
      det.width = coords[2];
      det.height = coords[3];
      break;
    }
    case BboxFormat::CXCYWH:
    default: {
      // [cx, cy, w, h] -> already in center format
      det.x = coords[0];
      det.y = coords[1];
      det.width = coords[2];
      det.height = coords[3];
      break;
    }
  }
}

std::vector<std::vector<SimpleDetection>> GenericPostprocessor::decode(
  const deep_ros::Tensor & output, const std::vector<ImageMeta> & metas) const
{
  std::vector<std::vector<SimpleDetection>> batch_detections;

  const auto & shape = output.shape();
  if (shape.empty()) {
    throw std::runtime_error("Output tensor has empty shape");
  }

  const float * data = output.data_as<float>();
  if (!data) {
    throw std::runtime_error("Output tensor has null data");
  }

  // Auto-detect layout if needed
  OutputLayout layout = layout_;
  if (layout.auto_detect) {
    layout = detectLayout(shape);
  }

  // Determine dimensions
  size_t batch_size = shape[layout.batch_dim];
  size_t num_detections = shape.size() > layout.detection_dim ? shape[layout.detection_dim] : 1;
  size_t num_features = shape.size() > layout.feature_dim ? shape[layout.feature_dim] : shape.back();

  batch_detections.reserve(std::min(batch_size, metas.size()));
  
  for (size_t b = 0; b < batch_size && b < metas.size(); ++b) {
    std::vector<SimpleDetection> dets;
    dets.reserve(num_detections);

    for (size_t d = 0; d < num_detections; ++d) {
      // For YOLO models: find the max class score (with sigmoid) from class logits
      // YOLOv8 format: [x, y, w, h, class0_score, class1_score, ..., classN_score]
      // No separate objectness score - use max class score as confidence
      
      float score = 0.0f;
      int32_t class_id = -1;
      
      if (num_features > layout.bbox_count) {
        // Find argmax of class scores (after sigmoid)
        float max_score = -std::numeric_limits<float>::max();
        size_t best_class = 0;
        
        size_t class_end = std::min(num_features, static_cast<size_t>(num_classes_ + layout.bbox_count));
        for (size_t c = layout.bbox_count; c < class_end; ++c) {
          float cls_logit = extractValue(data, b, d, c, shape);
          // Apply sigmoid: score = 1 / (1 + exp(-x))
          float cls_score = 1.0f / (1.0f + std::exp(-cls_logit));
          if (cls_score > max_score) {
            max_score = cls_score;
            best_class = c - layout.bbox_count;
          }
        }
        score = max_score;
        class_id = static_cast<int32_t>(best_class);
      } else if (layout.score_idx < num_features) {
        // Fallback: use explicit score index
        float raw_score = extractValue(data, b, d, layout.score_idx, shape);
        score = 1.0f / (1.0f + std::exp(-raw_score));  // Apply sigmoid
        if (layout.class_idx < num_features) {
          class_id = static_cast<int32_t>(std::round(extractValue(data, b, d, layout.class_idx, shape)));
        }
      }

      if (score < config_.score_threshold) {
        continue;
      }

      SimpleDetection det;
      convertBbox(data, b, d, shape, det);
      det.score = score;
      det.class_id = class_id;

      adjustToOriginal(det, metas[b], use_letterbox_);
      dets.push_back(det);
    }

    batch_detections.push_back(applyNms(dets, config_.nms_iou_threshold));
  }

  return batch_detections;
}

void GenericPostprocessor::adjustToOriginal(
  SimpleDetection & det, const ImageMeta & meta, bool use_letterbox) const
{
  float cx = det.x;
  float cy = det.y;
  float w = det.width;
  float h = det.height;

  cx -= meta.pad_x;
  cy -= meta.pad_y;

  if (use_letterbox) {
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

float GenericPostprocessor::iou(const SimpleDetection & a, const SimpleDetection & b)
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

std::vector<SimpleDetection> GenericPostprocessor::applyNms(
  std::vector<SimpleDetection> dets, float iou_threshold) const
{
  std::vector<SimpleDetection> result;
  if (dets.empty()) {
    return result;
  }

  std::stable_sort(
    dets.begin(), dets.end(), [](const SimpleDetection & a, const SimpleDetection & b) { 
      return a.score > b.score; 
    });

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
      if (iou(dets[i], dets[j]) > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

std::string GenericPostprocessor::classLabel(
  int class_id, const std::vector<std::string> & class_names) const
{
  if (class_id >= 0 && static_cast<size_t>(class_id) < class_names.size()) {
    return class_names[static_cast<size_t>(class_id)];
  }
  return std::to_string(class_id);
}

void GenericPostprocessor::fillDetectionMessage(
  const std_msgs::msg::Header & header,
  const std::vector<SimpleDetection> & detections,
  const ImageMeta & meta,
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
    d.label = classLabel(det.class_id, class_names_);
    out_msg.detections.push_back(d);
  }
#else
  (void)meta;
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
    hyp.hypothesis.class_id = classLabel(det.class_id, class_names_);
    hyp.hypothesis.score = det.score;
    d.results.push_back(hyp);
    out_msg.detections.push_back(d);
  }
#endif
}

}  // namespace deep_object_detection

