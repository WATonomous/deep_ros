# Multi-Camera Synchronization Node

A ROS2 node that uses message filters to time-synchronize N camera image messages, supporting both compressed and raw images.

## Features

- **Flexible camera count**: Supports 2-6 cameras simultaneously
- **Image format support**: Works with both `sensor_msgs/Image` (raw) and `sensor_msgs/CompressedImage`
- **Configurable synchronization**: Adjustable time tolerance and queue sizes
- **Real-time monitoring**: Synchronization statistics and rate monitoring
- **Easy integration**: Standard ROS2 node with parameter-based configuration

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_topics` | string[] | `[]` | List of camera image topics to synchronize |
| `camera_names` | string[] | `[]` | Names for cameras (auto-generated if empty) |
| `use_compressed` | bool | `false` | Use compressed images instead of raw RGB |
| `sync_tolerance_ms` | double | `50.0` | Max time difference for sync (milliseconds) |
| `queue_size` | int | `10` | Message filter queue size |
| `publish_sync_info` | bool | `true` | Publish synchronization statistics |

## Usage

### Parameter usage:
All configuration changes can be passed through CLI. Here are some example arguments:

```bash
  -p camera_topics:="['/camera1/image_raw', '/camera2/image_raw']" \
  -p camera_names:="['left_camera', 'right_camera']" \
  -p use_compressed:=true \
  -p sync_tolerance_ms:=30.0
```

## How It Works

The node uses ROS2 message filters with approximate time synchronization policy to match images from multiple cameras based on their timestamps. Key components:

1. **Message Subscribers**: Creates subscribers for each camera topic
2. **Synchronizer**: Uses `message_filters::sync_policies::ApproximateTime` to match messages
3. **Callback & Publishing**: Creates MultiImage msgs with timestamp to batch the camera images together
4. **Statistics**: Tracks synchronization rate and timing spread

### Synchronization Logic

- Messages are considered synchronized if their timestamps are within `sync_tolerance_ms`
- The synchronizer maintains a queue of recent messages from each camera
- When a valid sync set is found, all cameras' images are processed together
- Unmatched messages are eventually dropped when they exceed the age penalty

## Monitoring

The node provides several ways to monitor synchronization performance:

1. **Console logs**: Periodic sync rate and timing statistics
2. **ROS2 parameters**: Runtime inspection of configuration
3. **Debug output**: Detailed timestamp information (when debug logging enabled)

Example log output:

```
[INFO] [multi_camera_sync]: Sync #500: compressed images from 4 cameras, spread: 12.3 ms, rate: 29.8 Hz
```

## Dependencies

- `rclcpp` - ROS2 C++ client library
- `sensor_msgs` - Standard sensor message types
- `message_filters` - Time synchronization utilities
- `image_transport` - Efficient image transmission

## Limitations

- Maximum 6 cameras supported (can be extended by adding more sync policies)
- Uses approximate time synchronization (not exact)
- All cameras must use the same image message type (raw or compressed)
- Requires reasonably synchronized system clocks across camera sources
