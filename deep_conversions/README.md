# deep_conversions

Conversion utilities for seamlessly transforming ROS 2 sensor messages into tensors for machine learning inference and back to ROS messages for publishing results.

## Overview

`deep_conversions` provides high-performance, zero-copy conversions between ROS 2 sensor messages and the `deep_core` tensor system. This enables efficient integration of ML models with ROS 2 sensor data streams.

## Features

### 🔄 Bidirectional Conversions
- **ROS → Tensor**: Convert sensor messages to ML-ready tensors
- **Tensor → ROS**: Convert inference results back to ROS messages
- **Batch processing**: Handle multiple messages efficiently
- **Memory-efficient**: Minimal copying with custom allocators

### 📷 Comprehensive Sensor Support
- **Images**: All standard ROS image encodings
- **Point Clouds**: PointCloud2 with arbitrary field configurations  
- **Laser Scans**: Range and intensity data
- **IMU Data**: Orientation, acceleration, and angular velocity

### ⚡ Performance Optimized
- **Custom allocators**: Use backend-specific memory (CPU, GPU)
- **Zero-copy operations**: Direct memory mapping where possible
- **Batch processing**: Efficient multi-message tensor creation
- **Type safety**: Compile-time type checking and validation

## Supported Conversions

### Image Messages

Convert `sensor_msgs::msg::Image` to tensors with proper shape and data type handling:

```cpp
#include <deep_conversions/ros_conversions.hpp>

// Single image conversion
sensor_msgs::msg::Image ros_image;
auto allocator = get_gpu_allocator();  // Custom allocator
auto tensor = deep_ros::ros_conversions::from_image(ros_image, allocator);
// Shape: [1, height, width, channels]

// Batch conversion
std::vector<sensor_msgs::msg::Image> images;
auto batch_tensor = deep_ros::ros_conversions::from_image(images, allocator);
// Shape: [batch_size, height, width, channels]

// Convert back to ROS
sensor_msgs::msg::Image output_image;
deep_ros::ros_conversions::to_image(result_tensor, output_image, "bgr8");
```

### Point Cloud Messages

Convert `sensor_msgs::msg::PointCloud2` to structured tensors:

```cpp
// Point cloud conversion
sensor_msgs::msg::PointCloud2 cloud;
auto tensor = deep_ros::ros_conversions::from_pointcloud2(cloud, allocator);
// Shape: [num_points, num_fields] (e.g., [100000, 4] for XYZI)

// Access structured data
// Field order: [x, y, z, intensity, ...] as defined in cloud.fields
```

### Laser Scan Messages

Convert `sensor_msgs::msg::LaserScan` to range tensors:

```cpp
// Laser scan conversion
sensor_msgs::msg::LaserScan scan;
auto tensor = deep_ros::ros_conversions::from_laserscan(scan, allocator);
// Shape: [num_ranges] or [num_ranges, 2] if intensities present
```

### IMU Messages  

Convert `sensor_msgs::msg::Imu` to standardized tensors:

```cpp
// IMU conversion
sensor_msgs::msg::Imu imu;
auto tensor = deep_ros::ros_conversions::from_imu(imu, allocator);
// Shape: [10] -> [qx, qy, qz, qw, ax, ay, az, gx, gy, gz]
```

## Usage Examples

### Basic Image Processing Pipeline

```cpp
#include <deep_conversions/ros_conversions.hpp>
#include <deep_core/deep_node_base.hpp>

class ImageProcessorNode : public deep_ros::DeepNodeBase
{
public:
  ImageProcessorNode() : DeepNodeBase("image_processor") {
    // Load model and setup subscriptions
    load_plugin("onnxruntime_cpu");
    load_model("/models/object_detection.onnx");
    
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/camera/image", 10, 
      std::bind(&ImageProcessorNode::image_callback, this, std::placeholders::_1)
    );
    
    result_pub_ = create_publisher<sensor_msgs::msg::Image>("/detection/result", 10);
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      // Convert ROS image to tensor
      auto allocator = get_current_allocator();
      auto input_tensor = deep_ros::ros_conversions::from_image(*msg, allocator);
      
      // Run inference
      auto output_tensor = run_inference(input_tensor);
      
      // Convert result back to ROS image
      sensor_msgs::msg::Image result_msg;
      deep_ros::ros_conversions::to_image(output_tensor, result_msg, "bgr8", msg->header);
      
      // Publish result
      result_pub_->publish(result_msg);
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Processing failed: %s", e.what());
    }
  }
  
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr result_pub_;
};
```

### Multi-Sensor Fusion

```cpp
class FusionNode : public rclcpp::Node 
{
public:
  FusionNode() : Node("fusion_node") {
    // Setup synchronized subscribers
    image_sub_.subscribe(this, "/camera/image");
    lidar_sub_.subscribe(this, "/lidar/scan");
    imu_sub_.subscribe(this, "/imu/data");
    
    // Synchronize messages
    sync_.reset(new Synchronizer(SyncPolicy(10), image_sub_, lidar_sub_, imu_sub_));
    sync_->registerCallback(std::bind(&FusionNode::fusion_callback, this, _1, _2, _3));
  }

private:
  void fusion_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& image,
    const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan,
    const sensor_msgs::msg::Imu::ConstSharedPtr& imu) 
  {
    // Convert all sensors to tensors
    auto image_tensor = deep_ros::ros_conversions::from_image(*image, allocator_);
    auto lidar_tensor = deep_ros::ros_conversions::from_laserscan(*scan, allocator_);
    auto imu_tensor = deep_ros::ros_conversions::from_imu(*imu, allocator_);
    
    // Concatenate or process as needed for fusion model
    auto fused_input = create_fusion_tensor(image_tensor, lidar_tensor, imu_tensor);
    
    // Run fusion inference
    auto result = run_fusion_model(fused_input);
    
    // Publish results...
  }
  
  // Subscriber and synchronizer setup...
};
```

## Image Encoding Support

### Input Encodings (ROS → Tensor)
- **Grayscale**: `mono8`, `mono16`
- **Color**: `rgb8`, `bgr8`, `rgba8`, `bgra8`
- **High Dynamic Range**: `rgb16`, `bgr16`, `rgba16`, `bgra16`
- **YUV**: `yuv422`, `yuv444`
- **Bayer**: `bayer_rggb8`, `bayer_bggr8`, `bayer_gbrg8`, `bayer_grbg8`

### Output Encodings (Tensor → ROS)
- **Standard formats**: Automatic data type conversion
- **Normalization**: [0,1] float tensors → 8-bit integers
- **Multi-channel**: Support for arbitrary channel counts

### Automatic Type Conversion

```cpp
// The conversion system automatically handles:
ImageEncoding encoding_info = get_image_encoding_info("bgr8");
// encoding_info.dtype = DataType::UINT8
// encoding_info.channels = 3
// encoding_info.bytes_per_channel = 1
```

## Memory Management

### Custom Allocator Integration

```cpp
// Use backend-specific allocators
auto gpu_allocator = get_cuda_allocator();
auto cpu_allocator = get_cpu_allocator();

// GPU-based processing
auto gpu_tensor = deep_ros::ros_conversions::from_image(image, gpu_allocator);
auto gpu_result = run_gpu_inference(gpu_tensor);

// CPU-based processing  
auto cpu_tensor = deep_ros::ros_conversions::from_image(image, cpu_allocator);
auto cpu_result = run_cpu_inference(cpu_tensor);
```

### Zero-Copy Operations

```cpp
// When possible, conversions avoid memory copying:
// 1. Direct memory mapping for compatible formats
// 2. In-place transformations for layout changes
// 3. Shared memory for batch operations
```

## Performance Considerations

### Efficient Batch Processing

```cpp
// Batch multiple images for better throughput
std::vector<sensor_msgs::msg::Image> image_batch;
// ... collect images ...

// Single conversion call for entire batch
auto batch_tensor = deep_ros::ros_conversions::from_image(image_batch, allocator);
// Much faster than individual conversions
```

### Memory Layout Optimization

```cpp
// Tensors use optimal memory layouts:
// - Contiguous memory for cache efficiency
// - Proper alignment for SIMD operations  
// - Backend-specific memory locations (CPU/GPU)
```

## Error Handling

### Validation and Safety

```cpp
try {
  auto tensor = deep_ros::ros_conversions::from_image(image, allocator);
} catch (const std::invalid_argument& e) {
  // Unsupported encoding or invalid dimensions
  RCLCPP_ERROR(logger, "Conversion failed: %s", e.what());
} catch (const std::runtime_error& e) {
  // Memory allocation or data conversion errors
  RCLCPP_ERROR(logger, "Runtime error: %s", e.what());
}
```

### Common Issues

1. **Encoding mismatch**: Ensure output encoding matches tensor data type
2. **Memory alignment**: Some backends require specific memory alignment
3. **Batch size consistency**: All images in batch must have same dimensions

## Package Structure

```
deep_conversions/
├── include/
│   └── ros_conversions.hpp          # Main conversion interface
├── src/
│   └── ros_conversions.cpp          # Implementation
├── test/
│   └── test_ros_conversions.cpp     # Unit tests
└── CMakeLists.txt
```

## Dependencies

- **deep_core**: Tensor system and memory allocators
- **sensor_msgs**: ROS 2 sensor message types
- **std_msgs**: Standard ROS 2 message headers
- **rclcpp**: ROS 2 C++ client library

## Testing

```bash
# Run conversion tests
colcon test --packages-select deep_conversions

# Performance benchmarks
ros2 run deep_conversions benchmark_conversions
```

## Integration Examples

### With Computer Vision Pipeline

```cpp
// YOLOv8 object detection pipeline
auto image_tensor = from_image(ros_image, gpu_allocator);
auto detections = yolo_model->run_inference(image_tensor);
auto annotated_image = draw_detections(image_tensor, detections);
to_image(annotated_image, output_msg, "bgr8");
```

### With SLAM Systems

```cpp
// Visual-LiDAR SLAM
auto image_tensor = from_image(camera_msg, allocator);
auto lidar_tensor = from_pointcloud2(lidar_msg, allocator);
auto pose_estimate = slam_model->process(image_tensor, lidar_tensor);
```

## Future Enhancements

- [ ] **Compressed image support**: JPEG, PNG encoding/decoding
- [ ] **Advanced color spaces**: HSV, LAB, YCbCr conversions
- [ ] **Video streams**: Temporal tensor sequences
- [ ] **Point cloud compression**: Octree and voxel grid representations
- [ ] **Custom message types**: Extension framework for user-defined conversions

## License

Licensed under the Apache License, Version 2.0.