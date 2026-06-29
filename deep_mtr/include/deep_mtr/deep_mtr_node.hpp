#pragma once

#include <memory>
#include <string>

#include <deep_core/deep_node_base.hpp>
#include <deep_msgs/msg/mtr_prediction_array.hpp>
#include <deep_msgs/msg/mtr_scene.hpp>

namespace deep_mtr
{
class DeepMtrNode : public deep_ros::DeepNodeBase
{
public:
  explicit DeepMtrNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

protected:
  deep_ros::CallbackReturn on_configure_impl(const rclcpp_lifecycle::State &) override;
  deep_ros::CallbackReturn on_activate_impl(const rclcpp_lifecycle::State &) override;
  deep_ros::CallbackReturn on_deactivate_impl(const rclcpp_lifecycle::State &) override;
  deep_ros::CallbackReturn on_cleanup_impl(const rclcpp_lifecycle::State &) override;

private:
  void scene_callback(deep_msgs::msg::MtrScene::ConstSharedPtr scene);

  std::string input_topic_;
  std::string output_topic_;
  rclcpp::Subscription<deep_msgs::msg::MtrScene>::SharedPtr scene_sub_;
  rclcpp_lifecycle::LifecyclePublisher<deep_msgs::msg::MtrPredictionArray>::SharedPtr result_pub_;
};
}  // namespace deep_mtr
