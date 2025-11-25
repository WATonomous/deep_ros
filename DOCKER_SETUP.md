# Docker Container Setup Guide

This repository includes Docker container support for development. You can use either:
1. **VS Code Dev Containers** (recommended for development)
2. **Standalone Docker** (for CI/CD or headless use)

## Option 1: VS Code Dev Containers (Recommended)

### Prerequisites

- Docker installed and running
- VS Code with the "Dev Containers" extension installed

### Quick Start

1. **Open in VS Code**:
   ```bash
   code /workspaces/deep_ros
   ```

2. **Open in Container**:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type: `Dev Containers: Reopen in Container`
   - VS Code will build and start the container automatically

3. **The container includes**:
   - ROS 2 Humble pre-installed
   - All build tools (colcon, rosdep, etc.)
   - GPU support (if using GPU variant)
   - Workspace mounted at `/deep_ros_ws` (maps to your local workspace)

### Container Configuration

The devcontainer is configured in `.devcontainer/devcontainer.json`:

- **GPU Support**: Uses `nvidia/cuda` base image with TensorRT
- **Workspace Mount**: Your local code is mounted into the container
- **Auto-sourcing**: ROS 2 is automatically sourced in the container
- **VS Code Extensions**: C++, Python, CMake, ROS extensions pre-installed

### Building the Workspace in Container

Once inside the container:

```bash
# Navigate to workspace (already mounted)
cd /deep_ros_ws

# Install dependencies
rosdep install --from-paths . --ignore-src -r -y

# Build workspace
colcon build

# Source workspace
source install/setup.bash

# Set GPU library path (if using GPU)
export LD_LIBRARY_PATH="/deep_ros_ws/install/onnxruntime_gpu_vendor/lib:$LD_LIBRARY_PATH"
```

## Option 2: Standalone Docker Container

### Build the Docker Image

```bash
cd /workspaces/deep_ros

# Build GPU version
docker build \
  -f .devcontainer/Dockerfile \
  --build-arg ROS_DISTRO=humble \
  --build-arg TARGETARCH=gpu \
  --build-arg CUDA_VERSION=12.2.2 \
  --build-arg USERNAME=developer \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -t deep_ros:gpu \
  .

# Or build CPU version
docker build \
  -f .devcontainer/Dockerfile \
  --build-arg ROS_DISTRO=humble \
  --build-arg TARGETARCH=cpu \
  --build-arg USERNAME=developer \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -t deep_ros:cpu \
  .
```

### Run the Container

```bash
# GPU version
docker run -it --rm \
  --gpus all \
  --network host \
  -v /workspaces/deep_ros:/deep_ros_ws \
  -w /deep_ros_ws \
  deep_ros:gpu \
  bash

# CPU version
docker run -it --rm \
  --network host \
  -v /workspaces/deep_ros:/deep_ros_ws \
  -w /deep_ros_ws \
  deep_ros:cpu \
  bash
```

### Inside the Container

```bash
# Source ROS 2 (already in .bashrc, but verify)
source /opt/ros/humble/setup.bash

# Navigate to workspace
cd /deep_ros_ws

# Install dependencies
rosdep install --from-paths . --ignore-src -r -y

# Build workspace
colcon build

# Source workspace
source install/setup.bash

# For GPU support
export LD_LIBRARY_PATH="/deep_ros_ws/install/onnxruntime_gpu_vendor/lib:$LD_LIBRARY_PATH"
```

## Option 3: Docker Compose (Alternative)

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  deep_ros:
    build:
      context: .
      dockerfile: .devcontainer/Dockerfile
      args:
        ROS_DISTRO: humble
        TARGETARCH: gpu
        CUDA_VERSION: 12.2.2
        USERNAME: developer
        USER_UID: 1000
        USER_GID: 1000
    image: deep_ros:gpu
    container_name: deep_ros_dev
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - .:/deep_ros_ws
    working_dir: /deep_ros_ws
    network_mode: host
    stdin_open: true
    tty: true
    command: bash
```

Then run:
```bash
docker-compose up -d
docker-compose exec deep_ros bash
```

## Container Features

### What's Included

- **ROS 2 Humble**: Pre-installed and sourced
- **Build Tools**: gcc, g++, cmake, make
- **ROS Tools**: colcon, rosdep
- **Python**: Python 3 with pip
- **GPU Support**: CUDA runtime, TensorRT (GPU variant)
- **VS Code Extensions**: C++, Python, CMake, ROS (devcontainer only)

### Workspace Mounting

Your local workspace is mounted into the container at `/deep_ros_ws`. This means:
- ✅ Changes in the container are reflected locally
- ✅ Changes locally are reflected in the container
- ✅ Build artifacts (`build/`, `install/`, `log/`) persist locally

### GPU Support

For GPU support, ensure:
1. **NVIDIA Docker runtime** is installed:
   ```bash
   # Install nvidia-container-toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Verify GPU access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
   ```

## Development Workflow

### Daily Development

1. **Start container** (VS Code): Just open the folder, container starts automatically
2. **Make changes** to code
3. **Build**: `colcon build --packages-select <package>`
4. **Test**: Run your nodes/launch files
5. **Commit**: Changes are in your local git repo

### Rebuilding Container

If you modify the Dockerfile:

```bash
# In VS Code: Ctrl+Shift+P -> "Dev Containers: Rebuild Container"
# Or manually:
docker build -f .devcontainer/Dockerfile -t deep_ros:gpu .
```

## Troubleshooting

### Container Won't Start

- Check Docker is running: `docker ps`
- Check GPU access: `docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi`
- Check logs: `docker logs <container_name>`

### Build Errors in Container

- Ensure dependencies are installed: `rosdep install --from-paths . --ignore-src -r -y`
- Check ROS is sourced: `echo $ROS_DISTRO`
- Rebuild from scratch: `rm -rf build install log && colcon build`

### GPU Not Working

- Verify GPU access: `nvidia-smi` (should work inside container)
- Check library path: `echo $LD_LIBRARY_PATH`
- Verify CUDA: `nvcc --version` (if installed)

### Permission Issues

- Ensure user IDs match: `id -u` and `id -g` should match container user
- Fix permissions: `sudo chown -R $(id -u):$(id -g) /workspaces/deep_ros`

## Quick Reference

```bash
# Build workspace
colcon build

# Build specific package
colcon build --packages-select deep_object_detection

# Source workspace
source install/setup.bash

# Run node
ros2 launch deep_object_detection simple_detection.launch.yaml

# Check GPU
nvidia-smi

# Set GPU library path
export LD_LIBRARY_PATH="/deep_ros_ws/install/onnxruntime_gpu_vendor/lib:$LD_LIBRARY_PATH"
```

## Next Steps

- See `SETUP.md` for workspace setup details
- See `DEVELOPING.md` for development guidelines
- Check individual package READMEs for package-specific docs

