# MultiCameraInfo.msg
# Batched camera info for multiple cameras; order matches the corresponding MultiImage/MultiImageCompressed.
# Use header.stamp to associate with a multi_image message published at the same time.

std_msgs/Header header
sensor_msgs/CameraInfo[] camera_infos
