#include "deep_object_detection/tensorrt_inference.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace deep_object_detection {

// Logger implementation
void TensorRTInference::Logger::log(Severity severity, const char* msg) noexcept {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            std::cerr << "[TensorRT ERROR] " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cout << "[TensorRT WARNING] " << msg << std::endl;
            break;
        case Severity::kINFO:
            std::cout << "[TensorRT INFO] " << msg << std::endl;
            break;
        case Severity::kVERBOSE:
            // Suppress verbose logs by default
            break;
    }
}

TensorRTInference::TensorRTInference(const InferenceConfig& config)
    : config_(config)
    , initialized_(false)
    , gpu_input_buffer_(nullptr)
    , gpu_output_buffer_(nullptr)
    , cpu_input_buffer_(nullptr)
    , cpu_output_buffer_(nullptr)
    , input_size_(0)
    , output_size_(0)
    , input_binding_index_(-1)
    , output_binding_index_(-1) {
}

TensorRTInference::~TensorRTInference() {
    deallocateBuffers();
}

bool TensorRTInference::initialize() {
    if (initialized_) {
        return true;
    }

    if (!loadEngine()) {
        std::cerr << "Failed to load TensorRT engine" << std::endl;
        return false;
    }

    allocateBuffers();
    initialized_ = true;
    std::cout << "TensorRT inference engine initialized successfully" << std::endl;
    return true;
}

bool TensorRTInference::loadEngine() {
    std::ifstream file(config_.engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to read engine file: " << config_.engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), size));
    if (!engine_) {
        std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create TensorRT execution context" << std::endl;
        return false;
    }

    // Get binding indices
    input_binding_index_ = engine_->getBindingIndex(config_.input_blob_name.c_str());
    output_binding_index_ = engine_->getBindingIndex(config_.output_blob_name.c_str());

    if (input_binding_index_ == -1 || output_binding_index_ == -1) {
        std::cerr << "Failed to get binding indices" << std::endl;
        return false;
    }

    return true;
}

void TensorRTInference::allocateBuffers() {
    // Calculate buffer sizes
    auto input_dims = engine_->getBindingDimensions(input_binding_index_);
    auto output_dims = engine_->getBindingDimensions(output_binding_index_);

    input_size_ = config_.max_batch_size * 3 * config_.input_height * config_.input_width;
    
    // For YOLO-style output: [batch, num_detections, 85] (4 bbox + 1 conf + 80 classes)
    // Adjust based on your model's output format
    size_t num_classes = config_.class_names.size();
    size_t detection_size = 5 + num_classes; // x, y, w, h, conf + classes
    size_t max_detections = 25200; // Typical for YOLOv8 640x640
    output_size_ = config_.max_batch_size * max_detections * detection_size;

    // Allocate GPU memory
    cudaMalloc(&gpu_input_buffer_, input_size_ * sizeof(float));
    cudaMalloc(&gpu_output_buffer_, output_size_ * sizeof(float));

    // Allocate CPU memory
    cpu_input_buffer_ = new float[input_size_];
    cpu_output_buffer_ = new float[output_size_];

    std::cout << "Allocated buffers - Input size: " << input_size_ 
              << ", Output size: " << output_size_ << std::endl;
}

void TensorRTInference::deallocateBuffers() {
    if (gpu_input_buffer_) {
        cudaFree(gpu_input_buffer_);
        gpu_input_buffer_ = nullptr;
    }
    if (gpu_output_buffer_) {
        cudaFree(gpu_output_buffer_);
        gpu_output_buffer_ = nullptr;
    }
    if (cpu_input_buffer_) {
        delete[] cpu_input_buffer_;
        cpu_input_buffer_ = nullptr;
    }
    if (cpu_output_buffer_) {
        delete[] cpu_output_buffer_;
        cpu_output_buffer_ = nullptr;
    }
}

std::vector<std::vector<Detection>> TensorRTInference::inferBatch(const std::vector<cv::Mat>& images) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    
    if (!initialized_ || images.empty()) {
        return {};
    }

    int batch_size = std::min(static_cast<int>(images.size()), config_.max_batch_size);
    
    // Preprocess images
    std::vector<cv::Mat> preprocessed_images = preprocessBatch(
        std::vector<cv::Mat>(images.begin(), images.begin() + batch_size));
    
    // Copy preprocessed data to input buffer
    for (int i = 0; i < batch_size; ++i) {
        cv::Mat& img = preprocessed_images[i];
        float* input_ptr = cpu_input_buffer_ + i * 3 * config_.input_height * config_.input_width;
        
        // Convert BGR to RGB and normalize
        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);
        
        // Copy R, G, B channels
        for (int c = 0; c < 3; ++c) {
            float* channel_ptr = input_ptr + c * config_.input_height * config_.input_width;
            channels[2-c].convertTo(cv::Mat(config_.input_height, config_.input_width, CV_32F, channel_ptr), 
                                   CV_32F, 1.0f/255.0f);
        }
    }

    // Copy to GPU
    cudaMemcpy(gpu_input_buffer_, cpu_input_buffer_, 
               batch_size * 3 * config_.input_height * config_.input_width * sizeof(float), 
               cudaMemcpyHostToDevice);

    // Set batch size for dynamic shapes
    context_->setBindingDimensions(input_binding_index_, 
        nvinfer1::Dims4{batch_size, 3, config_.input_height, config_.input_width});

    // Run inference
    void* bindings[] = {gpu_input_buffer_, gpu_output_buffer_};
    bool success = context_->executeV2(bindings);
    
    if (!success) {
        std::cerr << "TensorRT inference failed" << std::endl;
        return {};
    }

    // Copy results back to CPU
    cudaMemcpy(cpu_output_buffer_, gpu_output_buffer_, 
               output_size_ * sizeof(float), cudaMemcpyDeviceToHost);

    // Post-process results
    return postprocessBatch(images, cpu_output_buffer_, batch_size);
}

std::vector<Detection> TensorRTInference::infer(const cv::Mat& image) {
    auto batch_results = inferBatch({image});
    return batch_results.empty() ? std::vector<Detection>{} : batch_results[0];
}

std::vector<cv::Mat> TensorRTInference::preprocessBatch(const std::vector<cv::Mat>& images) {
    std::vector<cv::Mat> preprocessed;
    preprocessed.reserve(images.size());
    
    for (const auto& image : images) {
        preprocessed.push_back(preprocessImage(image));
    }
    
    return preprocessed;
}

cv::Mat TensorRTInference::preprocessImage(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(config_.input_width, config_.input_height));
    return resized;
}

std::vector<std::vector<Detection>> TensorRTInference::postprocessBatch(
    const std::vector<cv::Mat>& original_images, 
    float* output_data, 
    int batch_size) {
    
    std::vector<std::vector<Detection>> batch_detections;
    batch_detections.reserve(batch_size);
    
    size_t single_output_size = output_size_ / config_.max_batch_size;
    
    for (int i = 0; i < batch_size; ++i) {
        int offset = i * single_output_size;
        auto detections = postprocessSingle(original_images[i], output_data, offset);
        batch_detections.push_back(applyNMS(detections));
    }
    
    return batch_detections;
}

std::vector<Detection> TensorRTInference::postprocessSingle(
    const cv::Mat& original_image, 
    float* output_data, 
    int output_offset) {
    
    std::vector<Detection> detections;
    
    // Calculate scale factors for coordinate conversion
    float scale_x = static_cast<float>(original_image.cols) / config_.input_width;
    float scale_y = static_cast<float>(original_image.rows) / config_.input_height;
    
    // Parse YOLO output format: [x, y, w, h, conf, class_probs...]
    size_t num_classes = config_.class_names.size();
    size_t detection_size = 5 + num_classes;
    size_t max_detections = 25200; // Adjust based on your model
    
    for (size_t i = 0; i < max_detections; ++i) {
        float* detection_ptr = output_data + output_offset + i * detection_size;
        
        float confidence = detection_ptr[4];
        if (confidence < config_.confidence_threshold) {
            continue;
        }
        
        // Find class with highest probability
        int best_class_id = 0;
        float best_class_prob = detection_ptr[5];
        for (size_t j = 1; j < num_classes; ++j) {
            if (detection_ptr[5 + j] > best_class_prob) {
                best_class_prob = detection_ptr[5 + j];
                best_class_id = j;
            }
        }
        
        float final_confidence = confidence * best_class_prob;
        if (final_confidence < config_.confidence_threshold) {
            continue;
        }
        
        // Convert from center format to top-left format and scale
        float center_x = detection_ptr[0] * scale_x;
        float center_y = detection_ptr[1] * scale_y;
        float width = detection_ptr[2] * scale_x;
        float height = detection_ptr[3] * scale_y;
        
        Detection det;
        det.x = center_x - width / 2;
        det.y = center_y - height / 2;
        det.width = width;
        det.height = height;
        det.confidence = final_confidence;
        det.class_id = best_class_id;
        det.class_name = best_class_id < config_.class_names.size() ? 
                        config_.class_names[best_class_id] : "unknown";
        
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<Detection> TensorRTInference::applyNMS(const std::vector<Detection>& detections) {
    std::vector<Detection> result;
    std::vector<Detection> sorted_detections = detections;
    
    // Sort by confidence
    std::sort(sorted_detections.begin(), sorted_detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<bool> suppressed(sorted_detections.size(), false);
    
    for (size_t i = 0; i < sorted_detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        result.push_back(sorted_detections[i]);
        
        // Suppress overlapping detections
        for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
            if (suppressed[j]) continue;
            
            // Calculate IoU
            float x1 = std::max(sorted_detections[i].x, sorted_detections[j].x);
            float y1 = std::max(sorted_detections[i].y, sorted_detections[j].y);
            float x2 = std::min(sorted_detections[i].x + sorted_detections[i].width,
                               sorted_detections[j].x + sorted_detections[j].width);
            float y2 = std::min(sorted_detections[i].y + sorted_detections[i].height,
                               sorted_detections[j].y + sorted_detections[j].height);
            
            if (x2 <= x1 || y2 <= y1) continue;
            
            float intersection = (x2 - x1) * (y2 - y1);
            float area1 = sorted_detections[i].width * sorted_detections[i].height;
            float area2 = sorted_detections[j].width * sorted_detections[j].height;
            float iou = intersection / (area1 + area2 - intersection);
            
            if (iou > config_.nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

} // namespace deep_object_detection