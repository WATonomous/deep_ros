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

## System Dependencies

If you encounter build errors related to missing build tools (e.g., `CMAKE_MAKE_PROGRAM is not set`, `CMAKE_C_COMPILER not set`), you may need to install the following system dependencies:

### Required Build Tools

Install the essential build tools package:

```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

The `build-essential` package includes:
- `gcc` - GNU C compiler
- `g++` - GNU C++ compiler
- `make` - Build automation tool
- `libc6-dev` - C library development files
- Other essential build tools

### Verifying Installation

After installation, verify the build tools are available:

```bash
which make && which gcc && which g++
```

You should see paths to all three tools. If any are missing, the `build-essential` package may not have installed correctly.

### Building the Project

Once build tools are installed, ensure ROS 2 is sourced and build the project:

```bash
source /opt/ros/humble/setup.bash  # or your ROS 2 distribution
colcon build --packages-up-to <package_name>
```

Or build all packages:

```bash
colcon build
```

### Stopping Containers

After using "Rebuild and Reopen in Container", you can stop containers using:

**Method 1: VS Code Command (Recommended)**
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type: `Dev Containers: Reopen Folder Locally`
- This closes the container and returns to local mode

**Method 2: Stop Container**
- Press `Ctrl+Shift+P`
- Type: `Dev Containers: Stop Container`
- This stops the container but keeps it available for later use

**Method 3: Using Docker Commands**
From a terminal (outside the container):
```bash
# List running containers
docker ps

# Stop a specific container by name
docker stop <container_name>

# Stop all dev containers
docker ps -q --filter "name=vsc-deep_ros" | xargs -r docker stop

# Stop and remove containers
docker stop <container_name>
docker rm <container_name>
```

**Method 4: Close VS Code Window**
- Simply closing the VS Code window will stop the container when you exit
- The container will remain stopped until you reopen the folder in container mode

### Restarting Containers

After stopping a container, you can restart it using:

**Method 1: VS Code Command (Recommended)**
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type: `Dev Containers: Reopen in Container`
- This will start the existing container or create a new one if needed

**Method 2: Rebuild and Reopen**
- Press `Ctrl+Shift+P`
- Type: `Dev Containers: Rebuild and Reopen in Container`
- Use this if you want to rebuild the container from scratch (e.g., after Dockerfile changes)

**Method 3: Using Docker Commands**
From a terminal (outside the container):
```bash
# List all containers (including stopped)
docker ps -a

# Start a stopped container by name
docker start <container_name>

# Start and attach to container
docker start <container_name>
docker attach <container_name>
```

**Method 4: Reopen Folder in VS Code**
- If VS Code detects a devcontainer configuration, it will prompt you to "Reopen in Container"
- Click the notification or use the command palette option

### Common Commands

Inside the container, you can do ros2 commands, colcon commands, rosdep, etc.
