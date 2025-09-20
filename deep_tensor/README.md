# deep_tensor

A lightweight, high-performance tensor library for ROS 2 deep learning applications. This package provides a generic tensor container with automatic memory management and seamless conversions between ROS sensor messages and tensor data structures.

## Features

- **Multi-dimensional tensor container** with support for various data types (float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool)
- **Automatic memory management** with both owned and borrowed memory patterns
- **ROS message conversions** for sensor_msgs (Image, PointCloud2, LaserScan, IMU)
- **Batch processing support** for efficient handling of multiple images
- **Zero-copy operations** where possible for optimal performance

## Basic Usage

Note: more usage patterns can be found in the tests directory.

### Creating Tensors

```cpp
#include "deep_tensor/tensor.hpp"

using namespace deep_ros;

// Create a 2D tensor (3x4) with float32 data
std::vector<size_t> shape = {3, 4};
Tensor tensor(shape, DataType::FLOAT32);

// Access raw data
float* data = tensor.data_as<float>();

// Get tensor properties
std::cout << "Shape: ";
for (auto dim : tensor.shape()) {
    std::cout << dim << " ";
}
std::cout << "\nRank: " << tensor.rank() << std::endl;
std::cout << "Total elements: " << tensor.size() << std::endl;
std::cout << "Bytes: " << tensor.byte_size() << std::endl;
```

### Working with Existing Data

```cpp
// Wrap existing data (non-owning)
float existing_data[12] = {1.0f, 2.0f, /* ... */};
std::vector<size_t> shape = {3, 4};
Tensor tensor(existing_data, shape, DataType::FLOAT32);
```

### Tensor Operations

```cpp
// Reshape tensor (total size must match)
auto reshaped = tensor.reshape({2, 6});

// Check if tensor is contiguous in memory
bool contiguous = tensor.is_contiguous();
```

## ROS Message Conversions

### Image Conversions

```cpp
#include "deep_tensor/ros_conversions.hpp"

using namespace deep_ros::ros_conversions;

// Convert single ROS Image to Tensor
sensor_msgs::msg::Image image_msg;
// ... populate image_msg ...

Tensor tensor = from_image(image_msg);
// Result shape: [1, height, width, channels] - always includes batch dimension

// Convert Tensor back to ROS Image
sensor_msgs::msg::Image output_image;
to_image(tensor, output_image, "rgb8");  // tensor must have batch size = 1

// With optional header
std_msgs::msg::Header header;
header.stamp = rclcpp::Clock().now();
to_image(tensor, output_image, "rgb8", header);
```

### Batch Image Processing

```cpp
// Convert multiple images to a batch tensor
std::vector<sensor_msgs::msg::Image> images;
// ... populate images ...

Tensor batch_tensor = from_image(images);  // Function overloading handles vector input
// Result shape: [batch_size, height, width, channels]

// Convert batch tensor back to vector of images
std::vector<sensor_msgs::msg::Image> output_images;
to_image(batch_tensor, output_images, "rgb8");  // Overload handles vector output
```

### Point Cloud Conversions

```cpp
// Convert PointCloud2 to Tensor
sensor_msgs::msg::PointCloud2 cloud_msg;
// ... populate cloud_msg ...

Tensor cloud_tensor = from_pointcloud2(cloud_msg);
// Result shape: [num_points, num_fields]
```

### Laser Scan Conversions

```cpp
// Convert LaserScan to Tensor
sensor_msgs::msg::LaserScan scan_msg;
// ... populate scan_msg ...

Tensor scan_tensor = from_laserscan(scan_msg);
// Result shape: [num_ranges] or [num_ranges, 2] if intensities present
```

### IMU Conversions

```cpp
// Convert IMU to Tensor
sensor_msgs::msg::Imu imu_msg;
// ... populate imu_msg ...

Tensor imu_tensor = from_imu(imu_msg);
// Result shape: [10] containing [qx,qy,qz,qw,ax,ay,az,gx,gy,gz]
```

## Supported Image Encodings

The library supports a wide range of ROS image encodings:

- **RGB/BGR formats**: `rgb8`, `rgba8`, `rgb16`, `rgba16`, `bgr8`, `bgra8`, `bgr16`, `bgra16`
- **Grayscale**: `mono8`, `mono16`
- **Bayer patterns**: `bayer_rggb8`, `bayer_bggr8`, `bayer_gbrg8`, `bayer_grbg8` (and 16-bit variants)
- **OpenCV formats**: `8UC1`, `8UC2`, `8UC3`, `8UC4`, `16SC1`, `32FC1`, `64FC1`, etc.
- **YUV formats**: `yuv422`, `YUV422_YUY2`, `UYVY`, `YUYV`

## Data Types

Supported tensor data types:

| DataType | C++ Type | Size (bytes) |
|----------|----------|--------------|
| FLOAT32  | float    | 4            |
| FLOAT64  | double   | 8            |
| INT8     | int8_t   | 1            |
| INT16    | int16_t  | 2            |
| INT32    | int32_t  | 4            |
| INT64    | int64_t  | 8            |
| UINT8    | uint8_t  | 1            |
| UINT16   | uint16_t | 2            |
| UINT32   | uint32_t | 4            |
| UINT64   | uint64_t | 8            |
| BOOL     | bool     | 1            |

## Integration in Your Package

### CMakeLists.txt

```cmake
find_package(deep_tensor REQUIRED)

target_link_libraries(your_target
  deep_tensor::deep_tensor_lib
)
```

### package.xml

```xml
<depend>deep_tensor</depend>
```

## Performance Considerations

- **Memory Management**: The library uses automatic memory management. Tensors created with the shape constructor own their memory, while tensors created with existing data pointers do not.
- **Zero-Copy**: When possible, conversions avoid unnecessary data copying.
- **Contiguous Memory**: Tensors maintain contiguous memory layout for optimal cache performance.
- **Move Semantics**: Full support for move construction and assignment to minimize copying.
