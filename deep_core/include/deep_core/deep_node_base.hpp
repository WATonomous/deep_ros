#pragma once

#include "plugin_interface.hpp"
#include "types/tensor.hpp"
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <memory>
#include <string>
#include <filesystem>

namespace deep_ros {

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

// Generic ROS lifecycle node that loads and manages plugins
class DeepNodeBase : public rclcpp_lifecycle::LifecycleNode {
public:
    explicit DeepNodeBase(
        const std::string& node_name,
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions()
    );
    
    virtual ~DeepNodeBase() = default;

protected:
    // Users override these for custom behavior
    virtual CallbackReturn on_configure_impl(const rclcpp_lifecycle::State& state) { 
        return CallbackReturn::SUCCESS; 
    }
    virtual CallbackReturn on_activate_impl(const rclcpp_lifecycle::State& state) { 
        return CallbackReturn::SUCCESS; 
    }
    virtual CallbackReturn on_deactivate_impl(const rclcpp_lifecycle::State& state) { 
        return CallbackReturn::SUCCESS; 
    }
    virtual CallbackReturn on_cleanup_impl(const rclcpp_lifecycle::State& state) { 
        return CallbackReturn::SUCCESS; 
    }
    virtual CallbackReturn on_shutdown_impl(const rclcpp_lifecycle::State& state) { 
        return CallbackReturn::SUCCESS; 
    }
    
    // Plugin management available to users
    bool load_plugin(const std::string& plugin_name);
    bool load_model(const std::filesystem::path& model_path);
    void unload_model();
    InferenceResult run_inference(std::vector<std::unique_ptr<Tensor>> inputs);
    
    // Plugin status
    bool is_plugin_loaded() const { return plugin_ != nullptr; }
    bool is_model_loaded() const { return model_loaded_; }
    std::string get_backend_name() const;

private:
    // Final lifecycle callbacks - base handles backend, then calls user impl
    CallbackReturn on_configure(const rclcpp_lifecycle::State& state) final;
    CallbackReturn on_activate(const rclcpp_lifecycle::State& state) final;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) final;
    CallbackReturn on_cleanup(const rclcpp_lifecycle::State& state) final;
    CallbackReturn on_shutdown(const rclcpp_lifecycle::State& state) final;
    
    // Plugin discovery and loading
    std::vector<std::string> discover_available_plugins();
    std::unique_ptr<InferencePluginInterface> load_plugin_library(const std::string& plugin_name);
    
    // State
    std::unique_ptr<InferencePluginInterface> plugin_;
    bool model_loaded_;
    std::string current_plugin_name_;
    std::filesystem::path current_model_path_;
    
    // ROS parameters
    void declare_parameters();
};

} // namespace deep_ros