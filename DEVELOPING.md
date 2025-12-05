# Development Guide

## Using Dev Containers

This project includes VS Code dev container configurations for easy ROS2 development across different distributions.

### Prerequisites

- Docker installed and running
- VS Code with the "Dev Containers" extension installed

### Getting Started

1. **Choose ROS distribution**:
   - Press `Ctrl+Shift+P` and run "Tasks: Run Task"
   - Select "Setup Devcontainer (CPU Only)" or "Setup Devcontainer (GPU)" and follow prompts

2. **Rebuild and open in container**:
   - Press `Ctrl+Shift+P` and run "Dev Containers: Rebuild and Reopen in Container" or the other variants
   - The container will automatically rebuild and reopen with your selected ROS version

### Container Features

- **Workspace**: Your code is mounted at `/deep_ros_ws`
- **ROS sourcing**: ROS is automatically sourced in your shell
- **Build tools**: Includes `colcon` and `rosdep` for ROS development
- **Extensions**: C++, CMake, Python, and XML support pre-installed

### Common Commands

Inside the container, you can do ros2 commands, colcon commands, rosdep, etc.

## Testing

### CI Testing (CPU Only)

```bash
export IS_CI=1
colcon build
source install/setup.bash && colcon test
colcon test-result --verbose
```

GPU backends are automatically skipped when `IS_CI=1`.

### Local GPU Testing

**Requires**: NVIDIA container with GPU access (Runtime Libraries and CuDNN)
Testing with GPU can be done using one of the support GPU devcontainers.

```bash
export IS_CI=0  # or unset IS_CI
colcon build
source install/setup.bash && colcon test
colcon test-result --verbose
```

### Testing Philosophy

Given that access to GPU runners are limited, we settled on a testing procedure to test GPU-related software locally. The codebase is designed such that **if a backend works with `deep_sample`, it works everywhere.**

- CPU backends: Tested in CI automatically
- GPU backends: Must be tested locally with GPU before release
- `deep_sample` validates all backend plugins end-to-end
- Other packages tested with CPU only in CI
