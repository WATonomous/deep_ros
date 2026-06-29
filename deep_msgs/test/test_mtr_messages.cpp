#include <gtest/gtest.h>

#include <deep_msgs/msg/map_polyline.hpp>
#include <deep_msgs/msg/mtr_prediction_array.hpp>
#include <deep_msgs/msg/mtr_scene.hpp>

TEST(MtrMessages, PreserveCorrelationAndSemanticMapData)
{
  deep_msgs::msg::MtrScene scene;
  scene.request_id = "request-7";
  scene.has_ego_pose = true;
  scene.has_map = true;

  deep_msgs::msg::MapPolyline centerline;
  centerline.lanelet_id = 42;
  centerline.semantic_type = deep_msgs::msg::MapPolyline::CENTERLINE;
  centerline.points.resize(2);
  scene.map_polylines.push_back(centerline);

  EXPECT_EQ(scene.request_id, "request-7");
  ASSERT_EQ(scene.map_polylines.size(), 1u);
  EXPECT_EQ(scene.map_polylines.front().lanelet_id, 42);
  EXPECT_EQ(scene.map_polylines.front().semantic_type, deep_msgs::msg::MapPolyline::CENTERLINE);

  deep_msgs::msg::MtrPredictionArray result;
  result.request_id = scene.request_id;
  EXPECT_EQ(result.request_id, scene.request_id);
}
