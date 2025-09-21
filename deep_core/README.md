# deep_core

Core abstractions for ML inference in ROS 2 lifecycle nodes.

## Overview

Provides:
- `Tensor`: Smart pointer for tensor data with custom memory allocators
- `DeepNodeBase`: Lifecycle node base class with plugin loading and optional bond support
- Plugin interfaces for backend inference engines and memory management

## Key Components

### Tensor
Multi-dimensional tensor smart pointer supporting:
- Custom memory allocators (CPU/GPU/aligned memory)
- View semantics (wrap existing data without copying)
- Standard tensor operations (reshape, data access)

### DeepNodeBase
Lifecycle node that handles:
- Dynamic backend plugin loading via pluginlib
- Model loading/unloading lifecycle
- Optional bond connections for nav2 integration
- Parameter-driven configuration

### Plugin Interfaces
Deep_ROS abstracts away hardware acceleration interfaces as plugins. This means that users have the
freedom to switch between different hardware accelerators at runtime. The backend plugin interface is
as follows:
- `DeepBackendPlugin`: Abstract interface for defining a backend plugin. Must implement:
  - `BackendMemoryAllocator`: Backend implementation for memory allocation and management
  - `BackendInferenceExecutor`: Backend implementation for running model inference

## Configuration

All nodes inherenting `deep_ros::DeepNodeBase` have the following settable parameters.

Required parameters:
- `Backend.plugin`: Plugin name (e.g., "onnxruntime_cpu")
- `model_path`: Path to model file (dynamically reconfigurable on runtime,
you can switch model's while the node is running!)

Optional parameters:
- `Bond.enable`: Enable bond connections (default: false)
- `Bond.bond_timeout`: Bond timeout in seconds (default: 4.0)
- `Bond.bond_heartbeat_period`: Heartbeat period in seconds (default: 0.1)
