#pragma once

#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

namespace deep_object_detection {

struct Detection {
    float x, y, width, height;  // Bounding box
    float confidence;           // Detection confidence
    int class_id;              // Class ID
    std::string class_name;    // Class name
};

struct InferenceConfig {
    std::string engine_path;
    std::vector<std::string> class_names;
    int input_width = 640;
    int input_height = 640;
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.4f;
    int max_batch_size = 8;
    bool use_fp16 = true;
    std::string input_blob_name = "images";
    std::string output_blob_name = "output0";
};

class TensorRTInference {
public:
    explicit TensorRTInference(const InferenceConfig& config);
    ~TensorRTInference();

    // Initialize the TensorRT engine
    bool initialize();
    
    // Batch inference for multiple images
    std::vector<std::vector<Detection>> inferBatch(const std::vector<cv::Mat>& images);
    
    // Single image inference (convenience method)
    std::vector<Detection> infer(const cv::Mat& image);
    
    // Get model info
    int getInputWidth() const { return config_.input_width; }
    int getInputHeight() const { return config_.input_height; }
    int getMaxBatchSize() const { return config_.max_batch_size; }
    bool isInitialized() const { return initialized_; }

private:
    // TensorRT related members
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    // CUDA memory management
    void* gpu_input_buffer_;
    void* gpu_output_buffer_;
    float* cpu_input_buffer_;
    float* cpu_output_buffer_;
    
    // Configuration and state
    InferenceConfig config_;
    bool initialized_;
    std::mutex inference_mutex_;
    
    // Model dimensions
    size_t input_size_;
    size_t output_size_;
    int input_binding_index_;
    int output_binding_index_;
    
    // Helper methods
    bool loadEngine();
    void allocateBuffers();
    void deallocateBuffers();
    std::vector<cv::Mat> preprocessBatch(const std::vector<cv::Mat>& images);
    cv::Mat preprocessImage(const cv::Mat& image);
    std::vector<std::vector<Detection>> postprocessBatch(
        const std::vector<cv::Mat>& original_images, 
        float* output_data, 
        int batch_size
    );
    std::vector<Detection> postprocessSingle(
        const cv::Mat& original_image, 
        float* output_data, 
        int output_offset
    );
    std::vector<Detection> applyNMS(const std::vector<Detection>& detections);
    
    // Logging helper
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };
    Logger logger_;
};

} // namespace deep_object_detection