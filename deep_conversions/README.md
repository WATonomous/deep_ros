# deep_conversions

ROS 2 sensor message to tensor conversion library.

## Overview

Provides generic conversions between ROS 2 sensor messages and `deep_core` tensors.

## Supported Conversions

### From ROS to Tensor
- **Images** (`sensor_msgs/msg/Image`) → Tensor with shape `[batch, height, width, channels]` or `[batch, height, width]` for grayscale
- **PointCloud2** (`sensor_msgs/msg/PointCloud2`) → Tensor with shape `[num_points, num_fields]`
- **LaserScan** (`sensor_msgs/msg/LaserScan`) → Tensor with shape `[num_ranges]` or `[num_ranges, 2]` with intensities
- **IMU** (`sensor_msgs/msg/Imu`) → Tensor with shape `[10]` containing quaternion, linear acceleration, and angular velocity

### From Tensor to ROS
- Tensor → **Image** with specified encoding

## Usage

Include the appropriate header for your message type:
- `deep_conversions/image_conversions.hpp`
- `deep_conversions/pointcloud_conversions.hpp`
- `deep_conversions/laserscan_conversions.hpp`
- `deep_conversions/imu_conversions.hpp`

All conversion functions are in the `deep_ros::ros_conversions` namespace and require a `BackendMemoryAllocator` for tensor creation.

## License

Copyright (c) 2025-present WATonomous. All rights reserved.

Licensed under the Apache License, Version 2.0.
