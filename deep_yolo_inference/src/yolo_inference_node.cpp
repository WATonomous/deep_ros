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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <deque>
#include <dlfcn.h>
#include <functional>
#include <fstream>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rmw/qos_profiles.h>
#include <rclcpp_components/register_node_macro.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <deep_core/types/tensor.hpp>
#include <deep_core/plugin_interfaces/deep_backend_plugin.hpp>
#include <deep_ort_backend_plugin/ort_backend_plugin.hpp>
#include <deep_ort_gpu_backend_plugin/ort_gpu_backend_plugin.hpp>

#include "deep_yolo_inference/detection_msg_alias.hpp"
#include "deep_yolo_inference/yolo_inference_node.hpp"

namespace deep_yolo_inference
{

namespace chrono = std::chrono;

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
  sensor_msgs::msg::Image::ConstSharedPtr msg;
  rclcpp::Time arrival_time;
};

struct PackedInput
{
  std::vector<float> data;
  std::vector<size_t> shape;
};

class ScopeExit
{
public:
  explicit ScopeExit(std::function<void()> fn) : fn_(std::move(fn)) {}
  ~ScopeExit()
  {
    if (fn_) {
      fn_();
    }
  }

private:
  std::function<void()> fn_;
};

class YoloInferenceNode : public rclcpp::Node
{
public:
  explicit YoloInferenceNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("yolo_inference_node", options)
  {
    declareAndReadParameters();
    validateParameters();

    loadClassNames();
    detection_pub_ = this->create_publisher<Detection2DArrayMsg>(params_.output_detections_topic, rclcpp::SystemDefaultsQoS{});

    multi_camera_mode_ = !params_.camera_topics.empty();
    auto transport = params_.input_transport;
    std::transform(
      transport.begin(), transport.end(), transport.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    rmw_qos_profile_t qos = rmw_qos_profile_sensor_data;
    qos.depth = static_cast<size_t>(params_.queue_size);
    qos.reliability = params_.input_qos_reliability == "reliable" ?
      RMW_QOS_POLICY_RELIABILITY_RELIABLE : RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;

    auto subscribe_with_transport = [this, &qos](const std::string & tr) {
      return image_transport::create_subscription(
        this,
        params_.input_image_topic,
        std::bind(&YoloInferenceNode::onImage, this, std::placeholders::_1),
        tr,
        qos);
    };

    if (multi_camera_mode_) {
      setupMultiCameraSubscriptions();
    } else {
      const bool topic_already_compressed = isCompressedTopic(params_.input_image_topic);
      if (params_.input_transport == "compressed" && topic_already_compressed) {
        auto qos_profile = rclcpp::SensorDataQoS();
        qos_profile.keep_last(params_.queue_size);
        if (params_.input_qos_reliability == "reliable") {
          qos_profile.reliable();
        } else {
          qos_profile.best_effort();
        }

        compressed_image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
          params_.input_image_topic,
          qos_profile,
          std::bind(&YoloInferenceNode::onCompressedImage, this, std::placeholders::_1));
        active_input_transport_ = "compressed_direct";
        RCLCPP_INFO(
          this->get_logger(),
          "Subscribing directly to compressed image topic: %s",
          params_.input_image_topic.c_str());
      } else {
        try {
          image_sub_ = subscribe_with_transport(transport);
          active_input_transport_ = transport;
        } catch (const std::exception & e) {
          if (transport != "raw") {
            RCLCPP_WARN(
              this->get_logger(),
              "Failed to create '%s' image transport subscription (%s); falling back to 'raw'",
              transport.c_str(),
              e.what());
            image_sub_ = subscribe_with_transport("raw");
            active_input_transport_ = "raw";
          } else {
            throw;
          }
        }
      }
    }

    batch_timer_ = this->create_wall_timer(
      chrono::milliseconds(5), std::bind(&YoloInferenceNode::onBatchTimer, this));

    buildProviderOrder();
    if (params_.model_path.empty()) {
      RCLCPP_WARN(this->get_logger(), "model_path is empty; backend initialization skipped.");
    } else if (!initializeBackend()) {
      throw std::runtime_error("Failed to initialize any execution provider");
    }

    RCLCPP_INFO(
      this->get_logger(),
      "YOLO inference node initialized. Model: %s, input size: %dx%d, batch limit: %d, provider: %s",
      params_.model_path.c_str(),
      params_.input_width,
      params_.input_height,
      params_.batch_size_limit,
      active_provider_.c_str());
  }

private:
  struct Params
  {
    std::string model_path;
    std::string input_image_topic{"/camera/image_raw"};
    std::vector<std::string> camera_topics;
    std::string topic_type{"compressed_image"};
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
    std::string preferred_provider{"tensorrt"};
    int device_id{0};
    bool warmup_tensor_shapes{true};
    bool enable_trt_engine_cache{false};
    std::string trt_engine_cache_path{"/tmp/deep_ros_ort_trt_cache"};
  };

  void declareAndReadParameters()
  {
    params_.model_path = this->declare_parameter<std::string>("model_path", "");
    params_.input_image_topic = this->declare_parameter<std::string>("input_image_topic", params_.input_image_topic);
    params_.camera_topics = this->declare_parameter<std::vector<std::string>>("camera_topics", params_.camera_topics);
    params_.topic_type = this->declare_parameter<std::string>("topic_type", params_.topic_type);
    params_.input_transport = this->declare_parameter<std::string>("input_transport", params_.input_transport);
    params_.input_qos_reliability =
      this->declare_parameter<std::string>("input_qos_reliability", params_.input_qos_reliability);
    params_.output_detections_topic =
      this->declare_parameter<std::string>("output_detections_topic", params_.output_detections_topic);
    params_.input_width = this->declare_parameter<int>("input_width", params_.input_width);
    params_.input_height = this->declare_parameter<int>("input_height", params_.input_height);
    params_.use_letterbox = this->declare_parameter<bool>("use_letterbox", params_.use_letterbox);
    params_.batch_size_limit = this->declare_parameter<int>("batch_size_limit", params_.batch_size_limit);
    params_.score_threshold = this->declare_parameter<double>("score_threshold", params_.score_threshold);
    params_.nms_iou_threshold = this->declare_parameter<double>("nms_iou_threshold", params_.nms_iou_threshold);
    params_.preferred_provider = this->declare_parameter<std::string>("preferred_provider", params_.preferred_provider);
    params_.device_id = this->declare_parameter<int>("device_id", params_.device_id);
    params_.class_names_path = this->declare_parameter<std::string>("class_names_path", params_.class_names_path);
    params_.queue_size = this->declare_parameter<int>("queue_size", params_.queue_size);
    params_.warmup_tensor_shapes =
      this->declare_parameter<bool>("warmup_tensor_shapes", params_.warmup_tensor_shapes);
    params_.enable_trt_engine_cache =
      this->declare_parameter<bool>("enable_trt_engine_cache", params_.enable_trt_engine_cache);
    params_.trt_engine_cache_path =
      this->declare_parameter<std::string>("trt_engine_cache_path", params_.trt_engine_cache_path);
  }

  void validateParameters()
  {
    auto fail = [this](const std::string & msg) {
      RCLCPP_ERROR(this->get_logger(), "%s", msg.c_str());
      throw std::runtime_error(msg);
    };

    auto normalize_transport = params_.input_transport;
    std::transform(
      normalize_transport.begin(), normalize_transport.end(), normalize_transport.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (normalize_transport != "raw" && normalize_transport != "compressed") {
      fail("input_transport must be either 'raw' or 'compressed'.");
    }
    params_.input_transport = normalize_transport;

    auto topic_type = params_.topic_type;
    std::transform(topic_type.begin(), topic_type.end(), topic_type.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (topic_type != "raw_image" && topic_type != "compressed_image") {
      fail("topic_type must be either 'raw_image' or 'compressed_image'.");
    }
    params_.topic_type = topic_type;

    auto reliability = params_.input_qos_reliability;
    std::transform(
      reliability.begin(), reliability.end(), reliability.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (reliability != "best_effort" && reliability != "reliable") {
      fail("input_qos_reliability must be either 'best_effort' or 'reliable'.");
    }
    params_.input_qos_reliability = reliability;

    if (params_.model_path.empty()) {
      fail("Parameter 'model_path' must be set to a valid ONNX model file.");
    } else {
      std::filesystem::path model_path(params_.model_path);
      if (!std::filesystem::exists(model_path)) {
        fail("Model path does not exist: " + params_.model_path);
      }
    }

    if (params_.batch_size_limit < 1 || params_.batch_size_limit > 6) {
      fail("batch_size_limit must be between 1 and 6.");
    }
    if (params_.batch_size_limit != 3) {
      RCLCPP_WARN(
        this->get_logger(),
        "batch_size_limit is fixed at 3; overriding requested value %d",
        params_.batch_size_limit);
      params_.batch_size_limit = 3;
    }
    if (params_.input_width <= 0 || params_.input_height <= 0) {
      fail("input_width and input_height must be positive.");
    }
    if (params_.device_id < 0) {
      fail("device_id must be non-negative.");
    }
    if (params_.score_threshold <= 0.0 || params_.score_threshold > 1.0) {
      fail("score_threshold must be in (0.0, 1.0].");
    }
    if (params_.nms_iou_threshold <= 0.0 || params_.nms_iou_threshold > 1.0) {
      fail("nms_iou_threshold must be in (0.0, 1.0].");
    }
    if (params_.queue_size < 1) {
      fail("queue_size must be at least 1.");
    }

    if (!params_.camera_topics.empty()) {
      if (params_.topic_type != "compressed_image") {
        fail("camera_topics currently requires topic_type 'compressed_image'.");
      }
    }

    auto normalize_pref = params_.preferred_provider;
    std::transform(normalize_pref.begin(), normalize_pref.end(), normalize_pref.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (normalize_pref != "tensorrt" && normalize_pref != "cuda" && normalize_pref != "cpu") {
      fail("preferred_provider must be one of: tensorrt, cuda, cpu.");
    }
  }

  void buildProviderOrder()
  {
    auto normalize_pref = params_.preferred_provider;
    std::transform(normalize_pref.begin(), normalize_pref.end(), normalize_pref.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });

    provider_order_.clear();
    if (normalize_pref == "cpu") {
      provider_order_.push_back(Provider::CPU);  // CPU only
    } else if (normalize_pref == "cuda") {
      provider_order_.push_back(Provider::CUDA);
      provider_order_.push_back(Provider::CPU);
    } else {
      provider_order_.push_back(Provider::TENSORRT);
      provider_order_.push_back(Provider::CUDA);
      provider_order_.push_back(Provider::CPU);
    }
    active_provider_index_ = 0;
  }

  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    enqueueImageWithTimestamp(msg, this->now());
  }

  void onCompressedImage(const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg)
  {
    handleCompressedImage(msg, -1);
  }

  void setupMultiCameraSubscriptions()
  {
    multi_camera_subscriptions_.clear();
    multi_camera_subscriptions_.reserve(params_.camera_topics.size());
    auto qos_depth = static_cast<size_t>(params_.queue_size);
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(qos_depth));
    if (params_.input_qos_reliability == "reliable") {
      qos_profile.reliable();
    } else {
      qos_profile.best_effort();
    }
    for (size_t i = 0; i < params_.camera_topics.size(); ++i) {
      const auto & topic = params_.camera_topics[i];
      auto sub = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        topic,
        qos_profile,
        [this, i](const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
          this->handleCompressedImage(msg, static_cast<int>(i));
        });
      multi_camera_subscriptions_.push_back(sub);
      RCLCPP_INFO(
        this->get_logger(),
        "Subscribed to compressed image topic %s (camera %zu)",
        topic.c_str(),
        i);
    }
    active_input_transport_ = "compressed_multi";
  }

  void handleCompressedImage(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg,
    int camera_id)
  {
    try {
      const auto now = this->now();
      cv::Mat compressed_mat(
        1,
        static_cast<int>(msg->data.size()),
        CV_8UC1,
        const_cast<unsigned char *>(msg->data.data()));
      cv::Mat decoded = cv::imdecode(compressed_mat, cv::IMREAD_COLOR);
      if (decoded.empty()) {
        RCLCPP_WARN(this->get_logger(), "Failed to decode compressed image; skipping frame");
        return;
      }

      cv_bridge::CvImage cv_image(msg->header, sensor_msgs::image_encodings::BGR8, decoded);
      auto image_msg = cv_image.toImageMsg();
      sensor_msgs::msg::Image::ConstSharedPtr image_const(image_msg);
      enqueueImageWithTimestamp(image_const, now);
      if (camera_id >= 0) {
        RCLCPP_DEBUG(
          this->get_logger(),
          "Enqueued frame from camera %d (topic %s)",
          camera_id,
          msg->header.frame_id.c_str());
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN(this->get_logger(), "Skipping compressed image due to error: %s", e.what());
    }
  }

  void enqueueImageWithTimestamp(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    const rclcpp::Time & arrival)
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    image_queue_.push_back({msg, arrival});
    const auto limit = queueLimit();
    if (limit > 0 && image_queue_.size() > limit) {
      image_queue_.pop_front();
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        5000,
        "Image queue exceeded limit (%zu); dropping oldest frame",
        limit);
    }
    RCLCPP_DEBUG(
      this->get_logger(),
      "Image received. Queue size: %zu, Stamp: %.6f",
      image_queue_.size(),
      rclcpp::Time(msg->header.stamp).seconds());
    RCLCPP_INFO_ONCE(this->get_logger(), "Received first image on %s", params_.input_image_topic.c_str());
  }

  size_t queueLimit() const
  {
    return params_.queue_size > 0 ? static_cast<size_t>(params_.queue_size) : 0;
  }

  bool isCompressedTopic(const std::string & topic) const
  {
    static const std::string slash_suffix{"/compressed"};
    static const std::string underscore_suffix{"_compressed"};

    if (topic.size() >= slash_suffix.size() &&
      topic.compare(topic.size() - slash_suffix.size(), slash_suffix.size(), slash_suffix) == 0)
    {
      return true;
    }

    if (topic.size() >= underscore_suffix.size() &&
      topic.compare(topic.size() - underscore_suffix.size(), underscore_suffix.size(), underscore_suffix) == 0)
    {
      return true;
    }

    return false;
  }

  std::string formatShape(const std::vector<size_t> & shape) const
  {
    std::string result;
    result.reserve(shape.size() * 4);
    for (size_t i = 0; i < shape.size(); ++i) {
      result += std::to_string(shape[i]);
      if (i + 1 < shape.size()) {
        result += ", ";
      }
    }
    return result;
  }

  bool isCudaRuntimeAvailable() const
  {
    const char * libs[] = {"libcudart.so.12", "libcudart.so.11", "libcudart.so", "libcuda.so.1", "libcuda.so"};
    for (const auto * lib : libs) {
      void * handle = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
      if (handle) {
        dlclose(handle);
        return true;
      }
    }
    return false;
  }

  void onBatchTimer()
  {
    if (processing_.exchange(true)) {
      return;
    }
    ScopeExit guard([this]() { processing_.store(false); });

    std::vector<QueuedImage> batch;
    const size_t required = static_cast<size_t>(params_.batch_size_limit);
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (image_queue_.size() < required) {
        return;
      }

      batch.reserve(required);
      for (size_t i = 0; i < required; ++i) {
        batch.push_back(image_queue_.front());
        image_queue_.pop_front();
      }
    }

    if (!batch.empty()) {
      processBatch(batch);
    }
  }

  void processBatch(const std::vector<QueuedImage> & batch)
  {
    if (!executor_) {
      RCLCPP_WARN(this->get_logger(), "No backend initialized; dropping batch");
      return;
    }
    try {
      RCLCPP_INFO(this->get_logger(), "Processing batch of size: %zu", batch.size());
      std::vector<cv::Mat> processed;
      std::vector<ImageMeta> metas;
      std::vector<std_msgs::msg::Header> headers;
      processed.reserve(batch.size());
      metas.reserve(batch.size());
      headers.reserve(batch.size());

      for (const auto & item : batch) {
        try {
          auto converted = cv_bridge::toCvCopy(item.msg, sensor_msgs::image_encodings::BGR8);
          ImageMeta meta;
          cv::Mat preprocessed = preprocessImage(converted->image, meta);
          processed.push_back(std::move(preprocessed));
          metas.push_back(meta);
          headers.push_back(item.msg->header);
          RCLCPP_DEBUG(
            this->get_logger(),
            "Preprocessed image size: %dx%d (orig %dx%d)",
            processed.back().cols,
            processed.back().rows,
            meta.original_width,
            meta.original_height);
        } catch (const std::exception & e) {
          RCLCPP_WARN(
            this->get_logger(),
            "Skipping image due to preprocessing error: %s",
            e.what());
        }
      }

      if (processed.empty()) {
        RCLCPP_WARN(this->get_logger(), "No valid images to process in this batch");
        return;
      }

      auto packed_input = packInput(processed);
      if (packed_input.data.empty()) {
        RCLCPP_WARN(this->get_logger(), "Packed input is empty; skipping inference");
        return;
      }

      RCLCPP_INFO(
        this->get_logger(),
        "Input tensor shape about to build: [%s]",
        formatShape(packed_input.shape).c_str());

      bool success = false;
      deep_ros::Tensor output_tensor;
      while (!success) {
        if (!executor_) {
          RCLCPP_ERROR(this->get_logger(), "No executor available; dropping batch");
          return;
        }

        // Build the input tensor inside a tight scope so destruction happens before any fallback swap.
        {
          auto input_tensor = buildInputTensor(packed_input);
          try {
            output_tensor = executor_->run_inference(input_tensor);
            success = true;
          } catch (const std::exception & e) {
            RCLCPP_WARN(
              this->get_logger(),
              "Inference failed on provider %s: %s",
              active_provider_.c_str(),
              e.what());
          }
        }

        if (!success) {
          if (!fallbackToNextProvider()) {
            RCLCPP_ERROR(this->get_logger(), "All backends failed. Dropping batch.");
            return;
          }
          // loop continues with new executor_/allocator_
        }
      }

      const auto & shape = output_tensor.shape();
      if (shape.size() >= 3) {
        RCLCPP_INFO(
          this->get_logger(),
          "Output tensor shape: [%zu, %zu, %zu]",
          shape[0], shape[1], shape[2]);
      } else {
        RCLCPP_WARN(this->get_logger(), "Unexpected output shape rank: %zu", shape.size());
      }

      auto batch_detections = decodeOutput(output_tensor, metas);
      for (size_t i = 0; i < batch_detections.size(); ++i) {
        RCLCPP_INFO(
          this->get_logger(),
          "Image %zu: %zu detections", i, batch_detections[i].size());
      }
      publishDetections(batch_detections, headers, metas);
      RCLCPP_INFO(this->get_logger(), "Published detections for %zu images", batch_detections.size());
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Batch processing failed: %s", e.what());
    }
  }

  cv::Mat preprocessImage(const cv::Mat & bgr, ImageMeta & meta) const
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
        resized,
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));

      meta.scale_x = meta.scale_y = scale;
      meta.pad_x = static_cast<float>(pad_left);
      meta.pad_y = static_cast<float>(pad_top);
    } else {
      cv::resize(bgr, resized, cv::Size(params_.input_width, params_.input_height));
      meta.scale_x = static_cast<float>(meta.original_width) / static_cast<float>(params_.input_width);
      meta.scale_y = static_cast<float>(meta.original_height) / static_cast<float>(params_.input_height);
      meta.pad_x = meta.pad_y = 0.0f;
    }

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat float_image;
    rgb.convertTo(float_image, CV_32F, 1.0 / 255.0);

    return float_image;
  }

  PackedInput packInput(const std::vector<cv::Mat> & images) const
  {
    PackedInput packed;
    if (images.empty()) {
      return packed;
    }

    const size_t batch = images.size();
    const size_t channels = 3;
    const size_t height = images[0].rows;
    const size_t width = images[0].cols;
    const size_t image_size = channels * height * width;

    packed.shape = {batch, channels, height, width};
    packed.data.resize(batch * image_size);

    for (size_t b = 0; b < batch; ++b) {
      const cv::Mat & img = images[b];
      if (img.channels() != 3 || img.type() != CV_32FC3) {
        throw std::runtime_error("Preprocessed image must be CV_32FC3");
      }
      const float * src = reinterpret_cast<const float *>(img.data);

      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          const size_t src_idx = (h * width + w) * channels;
          const size_t dst_base = b * image_size + h * width + w;
          packed.data[dst_base + 0 * height * width] = src[src_idx + 0];
          packed.data[dst_base + 1 * height * width] = src[src_idx + 1];
          packed.data[dst_base + 2 * height * width] = src[src_idx + 2];
        }
      }
    }

    return packed;
  }

  deep_ros::Tensor buildInputTensor(const PackedInput & packed) const
  {
    deep_ros::Tensor tensor(packed.shape, deep_ros::DataType::FLOAT32, allocator_);
    const size_t bytes = packed.data.size() * sizeof(float);
    allocator_->copy_from_host(tensor.data(), packed.data.data(), bytes);
    return tensor;
  }

  std::vector<std::vector<SimpleDetection>> decodeOutput(
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

    // Helper for already-decoded layout: [batch, num_detections, values]
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
          det.x = read_val(0);
          det.y = read_val(1);
          det.width = read_val(2);
          det.height = read_val(3);
          det.score = obj;
          det.class_id = static_cast<int32_t>(std::round(read_val(5)));

          adjustToOriginal(det, metas[b]);
          dets.push_back(det);
        }
        batch_detections.push_back(applyNms(dets));
      }
    };

    // Case 1: standard decoded detections [batch, num_det, 6+]
    if (dim2 <= 8) {
      decode_preboxed(dim1, dim2, false);
      return batch_detections;
    }

    // Case 2: swapped small-last-dim layout [batch, num_det, >8] is unlikely;
    // handle [batch, num_det, values] where values is in dim1.
    if (dim1 <= 8) {
      decode_preboxed(dim2, dim1, true);
      return batch_detections;
    }

    // Case 3: YOLOv8 raw output (channel-first anchors): [batch, channels, anchors] or [batch, anchors, channels]
    const bool channels_first = dim1 < dim2;
    const size_t channels = channels_first ? dim1 : dim2;
    const size_t anchors = channels_first ? dim2 : dim1;

    if (channels < 5) {
      throw std::runtime_error("Unsupported YOLO output layout; channels dimension too small");
    }

    // Determine where class scores start.
    // If we see more channels than classes+4, assume an explicit objectness at index 4.
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

  void adjustToOriginal(SimpleDetection & det, const ImageMeta & meta) const
  {
    // Input detections are in resized/letterboxed coordinates, with x/y as centers.
    float cx = det.x;
    float cy = det.y;
    float w = det.width;
    float h = det.height;

    // Remove padding then scale back
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

    // Convert center to top-left
    det.x = std::max(0.0f, cx - w * 0.5f);
    det.y = std::max(0.0f, cy - h * 0.5f);
    det.width = w;
    det.height = h;

    // Clamp to original image bounds
    det.x = std::min(det.x, static_cast<float>(meta.original_width));
    det.y = std::min(det.y, static_cast<float>(meta.original_height));
    det.width = std::min(det.width, static_cast<float>(meta.original_width) - det.x);
    det.height = std::min(det.height, static_cast<float>(meta.original_height) - det.y);
  }

  static float iou(const SimpleDetection & a, const SimpleDetection & b)
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

  std::vector<SimpleDetection> applyNms(std::vector<SimpleDetection> dets) const
  {
    // Group by class for per-class NMS
    std::vector<SimpleDetection> result;
    if (dets.empty()) {
      return result;
    }

    std::stable_sort(
      dets.begin(),
      dets.end(),
      [](const SimpleDetection & a, const SimpleDetection & b) { return a.score > b.score; });

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

  void publishDetections(
    const std::vector<std::vector<SimpleDetection>> & batch_detections,
    const std::vector<std_msgs::msg::Header> & headers,
    const std::vector<ImageMeta> & metas)
  {
    for (size_t i = 0; i < batch_detections.size() && i < headers.size() && i < metas.size(); ++i) {
      Detection2DArrayMsg msg;
      fillDetectionMessage(headers[i], batch_detections[i], metas[i], msg);
      detection_pub_->publish(msg);
    }
  }

  void fillDetectionMessage(
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
      const auto label = classLabel(det.class_id);
      (void)label;
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

  std::string classLabel(int class_id) const
  {
    if (class_id >= 0 && static_cast<size_t>(class_id) < class_names_.size()) {
      return class_names_[static_cast<size_t>(class_id)];
    }
    return std::to_string(class_id);
  }

  void warmupTensorShapeCache(Provider provider)
  {
    if (!params_.warmup_tensor_shapes) {
      return;
    }
    if (provider == Provider::CPU) {
      return;
    }
    if (!executor_ || !allocator_) {
      return;
    }
    if (params_.batch_size_limit < 1) {
      return;
    }

    const size_t channels = 3;
    const size_t height = static_cast<size_t>(params_.input_height);
    const size_t width = static_cast<size_t>(params_.input_width);
    const size_t per_image = channels * height * width;

    const int batch = params_.batch_size_limit;
    RCLCPP_INFO(
      this->get_logger(),
      "Priming %s backend tensor shapes for fixed batch size %d",
      providerToString(provider).c_str(),
      batch);

    PackedInput dummy;
    dummy.shape = {static_cast<size_t>(batch), channels, height, width};
    dummy.data.assign(static_cast<size_t>(batch) * per_image, 0.0f);

    auto input_tensor = buildInputTensor(dummy);
    try {
      (void)executor_->run_inference(input_tensor);
      RCLCPP_DEBUG(this->get_logger(), "Cached tensor shape for batch size %d", batch);
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        this->get_logger(),
        "Warmup inference for batch size %d failed: %s",
        batch,
        e.what());
    }
  }

  bool initializeBackend(size_t start_index = 0)
  {
    for (size_t idx = start_index; idx < provider_order_.size(); ++idx) {
      active_provider_index_ = idx;
      Provider provider = provider_order_[idx];

      if ((provider == Provider::TENSORRT || provider == Provider::CUDA) && !isCudaRuntimeAvailable()) {
        RCLCPP_WARN(
          this->get_logger(),
          "Skipping provider %s: CUDA runtime libraries not found (libcudart/libcuda).",
          providerToString(provider).c_str());
        continue;
      }

      try {
        switch (provider) {
          case Provider::TENSORRT:
          case Provider::CUDA: {
            const auto provider_name = providerToString(provider);
            auto overrides = std::vector<rclcpp::Parameter>{
              rclcpp::Parameter("Backend.device_id", params_.device_id),
              rclcpp::Parameter("Backend.execution_provider", provider_name),
              rclcpp::Parameter("Backend.trt_engine_cache_enable", params_.enable_trt_engine_cache),
              rclcpp::Parameter("Backend.trt_engine_cache_path", params_.trt_engine_cache_path)
            };
            auto backend_node = createBackendConfigNode(provider_name, std::move(overrides));
            auto plugin = std::make_shared<deep_ort_gpu_backend::OrtGpuBackendPlugin>();
            plugin->initialize(backend_node);
            allocator_ = plugin->get_allocator();
            executor_ = plugin->get_inference_executor();
            plugin_holder_ = plugin;
            break;
          }
          case Provider::CPU: {
            auto backend_node = createBackendConfigNode("cpu");
            auto plugin = std::make_shared<deep_ort_backend::OrtBackendPlugin>();
            plugin->initialize(backend_node);
            allocator_ = plugin->get_allocator();
            executor_ = plugin->get_inference_executor();
            plugin_holder_ = plugin;
            break;
          }
        }

        if (!executor_ || !allocator_) {
          throw std::runtime_error("Executor or allocator is null");
        }

        if (params_.model_path.empty()) {
          throw std::runtime_error("Model path is not set");
        }

        if (!executor_->load_model(params_.model_path)) {
          throw std::runtime_error("Failed to load model: " + params_.model_path);
        }

        active_provider_ = providerToString(provider);
        declareActiveProviderParameter(active_provider_);
        warmupTensorShapeCache(provider);

        RCLCPP_INFO(
          this->get_logger(),
          "Initialized backend using provider: %s (device %d)",
          active_provider_.c_str(),
          params_.device_id);
        return true;
      } catch (const std::exception & e) {
        RCLCPP_WARN(
          this->get_logger(),
          "Provider %s initialization failed: %s",
          providerToString(provider).c_str(),
          e.what());
      }
    }

    RCLCPP_ERROR(this->get_logger(), "Unable to initialize any execution provider");
    return false;
  }

  bool fallbackToNextProvider()
  {
    if (provider_order_.empty()) {
      return false;
    }

    const size_t next_index = active_provider_index_ + 1;
    if (next_index >= provider_order_.size()) {
      RCLCPP_ERROR(this->get_logger(), "No more providers to fall back to");
      return false;
    }

    active_provider_index_ = next_index;
    RCLCPP_WARN(
      this->get_logger(),
      "Falling back to provider: %s",
      providerToString(provider_order_[active_provider_index_]).c_str());
    return initializeBackend(active_provider_index_);
  }

  std::string providerToString(Provider provider) const
  {
    switch (provider) {
      case Provider::TENSORRT:
        return "tensorrt";
      case Provider::CUDA:
        return "cuda";
      case Provider::CPU:
        return "cpu";
      default:
        return "unknown";
    }
  }

  rclcpp_lifecycle::LifecycleNode::SharedPtr createBackendConfigNode(
    const std::string & suffix,
    std::vector<rclcpp::Parameter> overrides = {}) const
  {
    static std::atomic<uint64_t> backend_node_counter{0};
    rclcpp::NodeOptions options;
    if (!overrides.empty()) {
      options.parameter_overrides(overrides);
    }
    options.start_parameter_services(false);
    options.start_parameter_event_publisher(false);

    const auto node_id = backend_node_counter.fetch_add(1, std::memory_order_relaxed);
    auto node_name = "yolo_backend_" + suffix + "_" + std::to_string(node_id);
    return std::make_shared<rclcpp_lifecycle::LifecycleNode>(node_name, options);
  }

  void declareActiveProviderParameter(const std::string & value)
  {
    rcl_interfaces::msg::ParameterDescriptor desc;
    desc.read_only = true;
    desc.description = "Current execution provider in use (read-only)";

    if (this->has_parameter("active_provider")) {
      (void)this->set_parameters({rclcpp::Parameter("active_provider", value)});
    } else {
      this->declare_parameter<std::string>("active_provider", value, desc);
    }
  }

  void loadClassNames()
  {
    if (params_.class_names_path.empty()) {
      return;
    }

    class_names_.clear();

    std::ifstream file(params_.class_names_path);
    if (!file.is_open()) {
      RCLCPP_WARN(
        this->get_logger(),
        "Failed to open class_names_path: %s",
        params_.class_names_path.c_str());
      return;
    }

    std::string line;
    while (std::getline(file, line)) {
      if (!line.empty()) {
        class_names_.push_back(line);
      }
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Loaded %zu class names from %s",
      class_names_.size(),
      params_.class_names_path.c_str());
  }

  Params params_;

  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_image_sub_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr> multi_camera_subscriptions_;
  bool multi_camera_mode_{false};
  rclcpp::Publisher<Detection2DArrayMsg>::SharedPtr detection_pub_;
  rclcpp::TimerBase::SharedPtr batch_timer_;

  std::deque<QueuedImage> image_queue_;
  std::mutex queue_mutex_;
  std::atomic<bool> processing_{false};
  std::string active_input_transport_{"raw"};

  std::shared_ptr<deep_ros::BackendInferenceExecutor> executor_;
  std::shared_ptr<deep_ros::BackendMemoryAllocator> allocator_;
  std::shared_ptr<deep_ros::DeepBackendPlugin> plugin_holder_;

  std::vector<Provider> provider_order_;
  size_t active_provider_index_{0};
  std::string active_provider_{"unknown"};
  std::vector<std::string> class_names_;
};

std::shared_ptr<rclcpp::Node> createYoloInferenceNode(const rclcpp::NodeOptions & options)
{
  return std::make_shared<YoloInferenceNode>(options);
}

}  // namespace deep_yolo_inference

RCLCPP_COMPONENTS_REGISTER_NODE(deep_yolo_inference::YoloInferenceNode)
