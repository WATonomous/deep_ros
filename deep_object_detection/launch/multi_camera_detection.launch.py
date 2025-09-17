#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """
    Launch file for multi-camera object detection with batched inference
    Example usage:
    ros2 launch deep_object_detection multi_camera_detection.launch.py \
        model_engine_path:=/path/to/yolov8.engine \
        max_batch_size:=8
    """
    
    # Configuration
    config_file = PathJoinSubstitution([
        FindPackageShare('deep_object_detection'),
        'config',
        'object_detection_config.yaml'
    ])

    # Object detection node with multiple camera support
    object_detection_node = Node(
        package='deep_object_detection',
        executable='object_detection_node',
        name='multi_camera_object_detection',
        parameters=[
            config_file,
            {
                'model_engine_path': LaunchConfiguration('model_engine_path', default='/path/to/model.engine'),
                'camera_topics': [
                    '/camera1/image_raw',
                    '/camera2/image_raw', 
                    '/camera3/image_raw',
                    '/camera4/image_raw'
                ],
                'max_batch_size': LaunchConfiguration('max_batch_size', default=4),
                'inference_rate': LaunchConfiguration('inference_rate', default=30.0),
                'confidence_threshold': LaunchConfiguration('confidence_threshold', default=0.5),
                'nms_threshold': LaunchConfiguration('nms_threshold', default=0.4),
                'enable_visualization': LaunchConfiguration('enable_visualization', default=True),
                'enable_debug': LaunchConfiguration('enable_debug', default=False),
                'detection_topic': '/multi_camera/detections',
                'visualization_topic': '/multi_camera/detection_markers',
            }
        ],
        output='screen',
        emulate_tty=True,
    )

    # Declare launch arguments
    declare_model_path = DeclareLaunchArgument(
        'model_engine_path',
        default_value='/path/to/model.engine',
        description='Path to the TensorRT engine file'
    )
    
    declare_batch_size = DeclareLaunchArgument(
        'max_batch_size',
        default_value='4',
        description='Maximum batch size for inference'
    )
    
    declare_inference_rate = DeclareLaunchArgument(
        'inference_rate',
        default_value='30.0',
        description='Target inference rate in Hz'
    )
    
    declare_confidence = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Detection confidence threshold'
    )
    
    declare_nms = DeclareLaunchArgument(
        'nms_threshold',
        default_value='0.4',
        description='Non-maximum suppression threshold'
    )
    
    declare_visualization = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='Enable visualization markers'
    )
    
    declare_debug = DeclareLaunchArgument(
        'enable_debug',
        default_value='false',
        description='Enable debug output and performance monitoring'
    )

    return LaunchDescription([
        declare_model_path,
        declare_batch_size,
        declare_inference_rate,
        declare_confidence,
        declare_nms,
        declare_visualization,
        declare_debug,
        object_detection_node
    ])
