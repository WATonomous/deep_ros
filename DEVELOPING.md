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

Inside the container:

```bash
# Update apt
sudo apt update

# Update rosdep
rosdep update

# Install dependencies
rosdep install --from-paths . --ignore-src -r -y

# Build packages
colcon build

# Source workspace
source install/setup.bash
```

### Troubleshooting

- If the container fails to build, check that Docker is running
- To rebuild from scratch: "Dev Containers: Rebuild Container"
