# deep_core developing doc

## Design Principles

- Tensor is a smart pointer, not a traditional tensor class, it points to data in memory allocated by the backend memory allocator
- DeepNodeBase handles plugin loading automatically via parameters
- All backends are plugins - no hard framework dependencies
- Memory allocators enable zero-copy GPU integration

## Usage

### CMakeLists.txt

```CMakeLists.txt
find_package(deep_core REQUIRED)

target_link_libraries(${YOUR_LIBRARY}
    deep_core::deep_core_lib
)
```

### Creating an Inference Node

**Inherit from `DeepNodeBase`** - gets automatic plugin loading and model management

Key lifecycle callbacks to override:
- `on_configure_impl()` - Set up subscribers, publishers, services
- `on_activate_impl()` - Start processing (DeepNodeBase handles plugin/model loading)
- `on_deactivate_impl()` - Stop processing
- `on_cleanup_impl()` - Clean up resources

**DeepNodeBase automatically handles:**
- Loading backend plugin based on `Backend.plugin` parameter
- Loading model based on `model_path` parameter
- Bond connections if `Bond.enable` is true
- Calling your `*_impl()` methods after base functionality

**Your node just needs to:**
- Set up ROS interfaces (topics, services, actions)
- Process incoming data using `run_inference(Tensor)`
- Handle your specific business logic

Don't forget: `RCLCPP_COMPONENTS_REGISTER_NODE(your_namespace::YourNode)`

### Creating a Backend Plugin

1. **Implement three classes inheriting from:**

- `BackendMemoryAllocator` - Handle memory allocation/deallocation for your hardware
- `BackendInferenceExecutor` - Load models and run inference in your ML framework
- `DeepBackendPlugin` - Return instances of your allocator and executor

Key methods to implement:
- Allocator: `allocate()`, `deallocate()`, `allocator_type()`
- Executor: `load_model()`, `run_inference()`, `unload_model()`, `supported_model_formats()`
- Plugin: `backend_name()`, `get_allocator()`, `get_inference_executor()`

Don't forget: `PLUGINLIB_EXPORT_CLASS(YourPlugin, deep_ros::DeepBackendPlugin)`

1. **Create `plugins.xml`:**

```xml
<library path="my_backend_lib">
  <class name="my_backend" type="MyBackendPlugin" base_class_type="deep_ros::DeepBackendPlugin">
    <description>My custom backend</description>
  </class>
</library>
```
