#include "deep_mtr/deep_mtr_node.hpp"

#include <functional>
#include <utility>

namespace deep_mtr
{
DeepMtrNode::DeepMtrNode(const rclcpp::NodeOptions & options)
: DeepNodeBase("deep_mtr_node", options)
{
  declare_parameter("input_topic", "/mtr/scenes");
  declare_parameter("output_topic", "/mtr/predictions");
}

deep_ros::CallbackReturn DeepMtrNode::on_configure_impl(const rclcpp_lifecycle::State &)
{
  input_topic_ = get_parameter("input_topic").as_string();
  output_topic_ = get_parameter("output_topic").as_string();
  result_pub_ = create_publisher<deep_msgs::msg::MtrPredictionArray>(output_topic_, 10);
  return deep_ros::CallbackReturn::SUCCESS;
}

deep_ros::CallbackReturn DeepMtrNode::on_activate_impl(const rclcpp_lifecycle::State &)
{
  result_pub_->on_activate();
  scene_sub_ = create_subscription<deep_msgs::msg::MtrScene>(
    input_topic_, 10, std::bind(&DeepMtrNode::scene_callback, this, std::placeholders::_1));
  return deep_ros::CallbackReturn::SUCCESS;
}

deep_ros::CallbackReturn DeepMtrNode::on_deactivate_impl(const rclcpp_lifecycle::State &)
{
  scene_sub_.reset();
  result_pub_->on_deactivate();
  return deep_ros::CallbackReturn::SUCCESS;
}

deep_ros::CallbackReturn DeepMtrNode::on_cleanup_impl(const rclcpp_lifecycle::State &)
{
  scene_sub_.reset();
  result_pub_.reset();
  return deep_ros::CallbackReturn::SUCCESS;
}

void DeepMtrNode::scene_callback(deep_msgs::msg::MtrScene::ConstSharedPtr scene)
{
  RCLCPP_WARN_THROTTLE(
    get_logger(), *get_clock(), 5000,
    "MTR inference is not implemented; ignoring scene request '%s'", scene->request_id.c_str());
}
}  // namespace deep_mtr

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(deep_mtr::DeepMtrNode)
