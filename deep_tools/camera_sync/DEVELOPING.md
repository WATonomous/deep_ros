# Camera Sync Development Notes

## Known Issues & Limitations

### 1. Lifecycle Node Incompatibility (ROS Humble, Jazzy and Earlier)
**Problem**: Lifecycle nodes cannot subscribe to compressed images in ROS Humble and earlier distributions.
- **Root Cause**: Missing lifecycle support in `image_transport` for compressed image subscriptions
- **Solution**: There is a Macro used throughout the code which specifies if it is with the lifecycle node or not. This is crucial to keep in mind when developing as breaking changes can occur across ROS distros if it's not compatible with a Lifecycle Node or vice versa. It is best to test with both versions to ensure compatibility.

The Macro:

```cpp
#if ROS_DISTRO_NAME STREQUAL "rolling"
  add_definitions(-DUSE_LIFECYCLE_NODE=1)
#else
  add_definitions(-DUSE_LIFECYCLE_NODE=0)  // Disable for Humble/earlier
#endif
```

### 2. Message Filters Synchronization Limitation
**Problem**: `message_filters::Synchronizer` requires compile-time known number of topics.
- **Limitation**: Cannot dynamically sync N cameras - must handle 2-6 cameras with separate synchronizers
- **Implementation**: Switch statement creates different sync policies for each camera count

## Architecture Notes
- **Component-only**: No standalone executable - use component loading only which is ideal to allow for IPC (zero-copy)
- **Dual Mode**: Supports both raw and compressed image synchronization
