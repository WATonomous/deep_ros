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
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace deep_object_detection
{

GenericPostprocessor::GenericPostprocessor(
  const PostprocessingConfig & config,
  const OutputLayout & layout,
  const std::string & bbox_format,
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

  layout.batch_dim = 0;

  if (output_shape.size() == 2) {
    layout.detection_dim = 0;
    layout.feature_dim = 1;
    layout.bbox_start_idx = 0;
    layout.bbox_count = 4;
    layout.score_idx = 4;
    layout.class_idx = 5;
  } else if (output_shape.size() == 3) {
    size_t dim1 = output_shape[1];
    size_t dim2 = output_shape[2];

    if (dim1 < dim2) {
      layout.detection_dim = 2;
      layout.feature_dim = 1;
    } else {
      layout.detection_dim = 1;
      layout.feature_dim = 2;
    }

    layout.bbox_start_idx = 0;
    layout.bbox_count = 4;
    layout.score_idx = 4;
    layout.class_idx = 5;
  } else {
    layout.detection_dim = 1;
    layout.feature_dim = 2;
    layout.bbox_start_idx = 0;
    layout.bbox_count = 4;
    layout.score_idx = 4;
    layout.class_idx = 5;
  }

  return layout;
}

GenericPostprocessor::OutputLayout GenericPostprocessor::autoConfigure(
  const std::vector<size_t> & output_shape, const OutputLayoutConfig & layout_config)
{
  OutputLayout layout;

  if (!layout_config.auto_detect) {
    layout.auto_detect = false;
    layout.batch_dim = static_cast<size_t>(layout_config.batch_dim);
    layout.detection_dim = static_cast<size_t>(layout_config.detection_dim);
    layout.feature_dim = static_cast<size_t>(layout_config.feature_dim);
    layout.bbox_start_idx = static_cast<size_t>(layout_config.bbox_start_idx);
    layout.bbox_count = static_cast<size_t>(layout_config.bbox_count);
    layout.score_idx = static_cast<size_t>(layout_config.score_idx);
    layout.class_idx = (layout_config.class_idx >= 0) ? static_cast<size_t>(layout_config.class_idx) : SIZE_MAX;
    if (!output_shape.empty()) {
      layout.shape = output_shape;
    }
  } else if (!output_shape.empty()) {
    layout = detectLayout(output_shape);
  } else {
    layout.auto_detect = true;
  }

  return layout;
}

float GenericPostprocessor::applyActivation(float raw_score) const
{
  if (config_.score_activation == "sigmoid") {
    return 1.0f / (1.0f + std::exp(-raw_score));
  }
  // "softmax" and "none" return raw score
  return raw_score;
}

float GenericPostprocessor::extractValue(
  const float * data,
  size_t batch_idx,
  size_t detection_idx,
  size_t feature_idx,
  const std::vector<size_t> & shape) const
{
  if (shape.size() == 2) {
    return data[batch_idx * shape[1] + feature_idx];
  } else if (shape.size() == 3) {
    if (layout_.detection_dim == 1 && layout_.feature_dim == 2) {
      return data[(batch_idx * shape[1] + detection_idx) * shape[2] + feature_idx];
    } else if (layout_.detection_dim == 2 && layout_.feature_dim == 1) {
      return data[(batch_idx * shape[1] + feature_idx) * shape[2] + detection_idx];
    }
  }

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
  float coords[4];
  for (size_t i = 0; i < 4; ++i) {
    coords[i] = extractValue(bbox_data, batch_idx, detection_idx, layout_.bbox_start_idx + i, shape);
  }

  if (bbox_format_ == "xyxy") {
    det.x = (coords[0] + coords[2]) * 0.5f;
    det.y = (coords[1] + coords[3]) * 0.5f;
    det.width = coords[2] - coords[0];
    det.height = coords[3] - coords[1];
  } else if (bbox_format_ == "xywh") {
    det.x = coords[0] + coords[2] * 0.5f;
    det.y = coords[1] + coords[3] * 0.5f;
    det.width = coords[2];
    det.height = coords[3];
  } else {
    // Default: cxcywh format
    det.x = coords[0];
    det.y = coords[1];
    det.width = coords[2];
    det.height = coords[3];
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

  OutputLayout layout = layout_;
  if (layout.auto_detect) {
    layout = detectLayout(shape);
  }

  size_t batch_size = shape[layout.batch_dim];
  size_t num_detections = shape.size() > layout.detection_dim ? shape[layout.detection_dim] : 1;
  size_t num_features = shape.size() > layout.feature_dim ? shape[layout.feature_dim] : shape.back();

  batch_detections.reserve(std::min(batch_size, metas.size()));

  for (size_t b = 0; b < batch_size && b < metas.size(); ++b) {
    std::vector<SimpleDetection> dets;
    dets.reserve(num_detections);

    for (size_t d = 0; d < num_detections; ++d) {
      float score = 0.0f;
      int32_t class_id = -1;

      size_t class_start_idx =
        (config_.class_score_start_idx >= 0) ? static_cast<size_t>(config_.class_score_start_idx) : layout.bbox_count;
      size_t class_count = (config_.class_score_count > 0) ? static_cast<size_t>(config_.class_score_count)
                                                           : static_cast<size_t>(num_classes_);

      if (config_.class_score_mode == "all_classes" && num_features > class_start_idx) {
        size_t class_end = std::min(num_features, class_start_idx + class_count);
        std::vector<float> class_logits;
        class_logits.reserve(class_end - class_start_idx);

        for (size_t c = class_start_idx; c < class_end; ++c) {
          float cls_logit = extractValue(data, b, d, c, shape);
          class_logits.push_back(cls_logit);
        }

        if (config_.score_activation == "softmax") {
          float max_logit = *std::max_element(class_logits.begin(), class_logits.end());
          float sum_exp = 0.0f;
          for (float & logit : class_logits) {
            logit = std::exp(logit - max_logit);
            sum_exp += logit;
          }

          float max_score = -std::numeric_limits<float>::max();
          size_t best_class = 0;
          for (size_t i = 0; i < class_logits.size(); ++i) {
            float prob = class_logits[i] / sum_exp;
            if (prob > max_score) {
              max_score = prob;
              best_class = i;
            }
          }
          score = max_score;
          class_id = static_cast<int32_t>(best_class);
        } else {
          float max_score = -std::numeric_limits<float>::max();
          size_t best_class = 0;
          for (size_t i = 0; i < class_logits.size(); ++i) {
            float cls_score = applyActivation(class_logits[i]);
            if (cls_score > max_score) {
              max_score = cls_score;
              best_class = i;
            }
          }
          score = max_score;
          class_id = static_cast<int32_t>(best_class);
        }
      } else if (config_.class_score_mode == "single_confidence") {
        if (layout.score_idx < num_features) {
          float raw_score = extractValue(data, b, d, layout.score_idx, shape);
          score = applyActivation(raw_score);
        }
        if (layout.class_idx < num_features && layout.class_idx != SIZE_MAX) {
          class_id = static_cast<int32_t>(std::round(extractValue(data, b, d, layout.class_idx, shape)));
        }
      } else if (layout.score_idx < num_features) {
        float raw_score = extractValue(data, b, d, layout.score_idx, shape);
        score = applyActivation(raw_score);
        if (layout.class_idx < num_features && layout.class_idx != SIZE_MAX) {
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

    if (config_.enable_nms) {
      batch_detections.push_back(applyNms(dets, config_.nms_iou_threshold));
    } else {
      batch_detections.push_back(dets);
    }
  }

  return batch_detections;
}

std::vector<std::vector<SimpleDetection>> GenericPostprocessor::decodeMultiOutput(
  const std::vector<deep_ros::Tensor> & outputs, const std::vector<ImageMeta> & metas) const
{
  std::vector<std::vector<SimpleDetection>> batch_detections;

  if (outputs.empty()) {
    throw std::runtime_error("No output tensors provided");
  }

  const int boxes_idx = config_.output_boxes_idx;
  const int scores_idx = config_.output_scores_idx;
  const int classes_idx = config_.output_classes_idx;

  if (boxes_idx < 0 || boxes_idx >= static_cast<int>(outputs.size())) {
    throw std::runtime_error("Invalid output_boxes_idx: " + std::to_string(boxes_idx));
  }
  if (scores_idx < 0 || scores_idx >= static_cast<int>(outputs.size())) {
    throw std::runtime_error("Invalid output_scores_idx: " + std::to_string(scores_idx));
  }

  const auto & boxes_tensor = outputs[boxes_idx];
  const auto & scores_tensor = outputs[scores_idx];

  const auto & boxes_shape = boxes_tensor.shape();
  const auto & scores_shape = scores_tensor.shape();

  if (boxes_shape.empty() || scores_shape.empty()) {
    throw std::runtime_error("Output tensors have empty shapes");
  }

  const float * boxes_data = boxes_tensor.data_as<float>();
  const float * scores_data = scores_tensor.data_as<float>();

  if (!boxes_data || !scores_data) {
    throw std::runtime_error("Output tensors have null data");
  }

  size_t batch_size = boxes_shape[0];
  size_t num_detections = boxes_shape.size() > 1 ? boxes_shape[1] : 1;
  size_t bbox_dims = boxes_shape.size() > 2 ? boxes_shape[2] : boxes_shape.back();

  if (bbox_dims < 4) {
    throw std::runtime_error("Boxes tensor must have at least 4 values per detection");
  }

  batch_detections.reserve(std::min(batch_size, metas.size()));

  for (size_t b = 0; b < batch_size && b < metas.size(); ++b) {
    std::vector<SimpleDetection> dets;
    dets.reserve(num_detections);

    for (size_t d = 0; d < num_detections; ++d) {
      size_t box_offset = (b * num_detections + d) * bbox_dims;
      float x = boxes_data[box_offset];
      float y = boxes_data[box_offset + 1];
      float w = boxes_data[box_offset + 2];
      float h = boxes_data[box_offset + 3];

      float score = 0.0f;
      int32_t class_id = -1;

      size_t scores_dim1 = scores_shape.size() > 1 ? scores_shape[1] : 1;
      size_t scores_dim2 = scores_shape.size() > 2 ? scores_shape[2] : 1;

      if (scores_dim1 == static_cast<size_t>(num_classes_) || scores_dim2 == static_cast<size_t>(num_classes_)) {
        bool detections_first = (scores_dim1 == num_detections);

        if (detections_first) {
          size_t score_base = (b * num_detections + d) * num_classes_;
          float max_score = -std::numeric_limits<float>::max();
          size_t best_class = 0;

          for (size_t c = 0; c < static_cast<size_t>(num_classes_); ++c) {
            float raw_score = scores_data[score_base + c];
            float activated_score = applyActivation(raw_score);
            if (activated_score > max_score) {
              max_score = activated_score;
              best_class = c;
            }
          }
          score = max_score;
          class_id = static_cast<int32_t>(best_class);
        } else {
          // [batch, num_classes, num_detections]
          float max_score = -std::numeric_limits<float>::max();
          size_t best_class = 0;

          for (size_t c = 0; c < static_cast<size_t>(num_classes_); ++c) {
            float raw_score = scores_data[b * num_classes_ * num_detections + c * num_detections + d];
            float activated_score = applyActivation(raw_score);
            if (activated_score > max_score) {
              max_score = activated_score;
              best_class = c;
            }
          }
          score = max_score;
          class_id = static_cast<int32_t>(best_class);
        }
      } else {
        size_t score_offset = b * num_detections + d;
        float raw_score = scores_data[score_offset];
        score = applyActivation(raw_score);

        if (classes_idx >= 0 && classes_idx < static_cast<int>(outputs.size())) {
          const auto & classes_tensor = outputs[classes_idx];
          const auto & classes_shape = classes_tensor.shape();
          const float * classes_data = classes_tensor.data_as<float>();
          if (classes_data && classes_shape.size() >= 2) {
            size_t class_offset = b * classes_shape[1] + d;
            class_id = static_cast<int32_t>(std::round(classes_data[class_offset]));
          }
        }
      }

      if (score < config_.score_threshold) {
        continue;
      }

      SimpleDetection det;
      if (bbox_format_ == "cxcywh") {
        det.x = x;
        det.y = y;
        det.width = w;
        det.height = h;
      } else if (bbox_format_ == "xyxy") {
        det.x = x;
        det.y = y;
        det.width = w - x;
        det.height = h - y;
      } else if (bbox_format_ == "xywh") {
        det.x = x;
        det.y = y;
        det.width = w;
        det.height = h;
      }

      det.score = score;
      det.class_id = class_id;

      adjustToOriginal(det, metas[b], use_letterbox_);
      dets.push_back(det);
    }

    if (config_.enable_nms) {
      batch_detections.push_back(applyNms(dets, config_.nms_iou_threshold));
    } else {
      batch_detections.push_back(dets);
    }
  }

  return batch_detections;
}

void GenericPostprocessor::adjustToOriginal(SimpleDetection & det, const ImageMeta & meta, bool use_letterbox) const
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
      if (iou(dets[i], dets[j]) > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

std::string GenericPostprocessor::classLabel(int class_id, const std::vector<std::string> & class_names) const
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
}

}  // namespace deep_object_detection
