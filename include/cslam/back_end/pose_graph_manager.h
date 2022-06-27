#ifndef _POSEGRAPHMANAGER_H_
#define _POSEGRAPHMANAGER_H_

#include <rclcpp/rclcpp.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/LabeledSymbol.h>
#include <gtsam/linear/NoiseModel.h>
 
#include <cslam_common_interfaces/msg/keyframe_odom.hpp>

class PoseGraphManager {
public:
  /**
   * @brief Initialization of parameters and ROS 2 objects
   *
   * @param node ROS 2 node handle
   */
  PoseGraphManager(std::shared_ptr<rclcpp::Node> &node);
  ~PoseGraphManager(){};

  /**
   * @brief Converts odometry message to gtsam::Pose3
   * 
   * @param odom_msg Odometry message
   * @param pose Pose data
   */
  void odometry_msg_to_pose3(const nav_msgs::msg::Odometry& odom_msg, gtsam::Pose3& pose);

  /**
   * @brief Receives odometry msg + keyframe id
   * 
   * @param msg 
   */
  void odometry_callback(const cslam_common_interfaces::msg::KeyframeOdom::ConstSharedPtr msg);

private:

  // TODO: document
  std::shared_ptr<rclcpp::Node> node_;

  unsigned int nb_robots_, robot_id_;

  unsigned char graph_label_, robot_label_;

  int pose_graph_manager_process_period_ms_;

  gtsam::SharedNoiseModel default_noise_model_;
  float rotation_default_noise_std_, translation_default_noise_std_;

  gtsam::NonlinearFactorGraph::shared_ptr pose_graph_;
  gtsam::Values::shared_ptr current_pose_estimates_;
  gtsam::Pose3 latest_local_pose_;
  gtsam::LabeledSymbol latest_local_symbol_;

  rclcpp::Subscription<
      cslam_common_interfaces::msg::KeyframeOdom>::SharedPtr
      odometry_subscriber_;

};

#endif