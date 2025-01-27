#include "cslam/front_end/stereo_handler.h"
#include <rtabmap_conversions/MsgConversion.h>
#include "tf2_eigen/tf2_eigen.hpp"

using namespace rtabmap;
using namespace cslam;

template class cslam::StereoHandler<ApproximateStereoSync>;
template class cslam::StereoHandler<ExactStereoSync>;

template<typename SyncPolicy>
StereoHandler<SyncPolicy>::StereoHandler(rclcpp::Node * node) : SensorHandler(node) {
  node_->declare_parameter<std::string>("frontend.left_image_topic", "left/image_rect");
  node_->declare_parameter<std::string>("frontend.right_image_topic", "right/image_rect");
  node_->declare_parameter<std::string>("frontend.left_camera_info_topic",
                                       "left/camera_info");
  node_->declare_parameter<std::string>("frontend.right_camera_info_topic",
                                       "right/camera_info");

  auto qos = rclcpp::SensorDataQoS().get_rmw_qos_profile();
  // Subscriber for stereo images
  sub_image_rect_left_.subscribe(
      node_, node_->get_parameter("frontend.left_image_topic").as_string(), "raw",
      qos);
  sub_image_rect_right_.subscribe(
      node_, node_->get_parameter("frontend.right_image_topic").as_string(), "raw",
      qos);
  sub_camera_info_left_.subscribe(
      node_, node_->get_parameter("frontend.left_camera_info_topic").as_string(),
      qos);
  sub_camera_info_right_.subscribe(
      node_, node_->get_parameter("frontend.right_camera_info_topic").as_string(),
      qos);

  stereo_synchronizer = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(max_queue_size_), sub_image_rect_left_, sub_image_rect_right_,
        sub_camera_info_left_, sub_camera_info_right_);
    if constexpr (std::is_same_v<SyncPolicy, ApproximateStereoSync>) {
      stereo_synchronizer->getPolicy()->setMaxIntervalDuration(rclcpp::Duration::from_seconds(node_->get_parameter("frontend.sync_period").as_double()));
    }
  //stereo_synchronizer->getPolicy()->setMaxIntervalDuration(rclcpp::Duration::from_seconds(node_->get_parameter("frontend.sync_period").as_double()));
  stereo_synchronizer->registerCallback(
        std::bind(&StereoHandler::stereo_callback, this, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3,
                    std::placeholders::_4));

    }
template<typename SyncPolicy>
void StereoHandler<SyncPolicy>::stereo_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr image_rect_left,
    const sensor_msgs::msg::Image::ConstSharedPtr image_rect_right,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_left,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_right) {

  auto sensor_data = std::make_shared<rtabmap_msgs::msg::SensorData>();
  sensor_data->left = *image_rect_left;
  sensor_data->right = *image_rect_right;
  sensor_data->header.stamp = image_rect_left->header.stamp;
  sensor_data->left_camera_info.push_back(*camera_info_left);
  sensor_data->right_camera_info.push_back(*camera_info_right);
  if(base_frame_id_.length() > 0) {
      geometry_msgs::msg::TransformStamped t;

      try {
        t = tf_buffer_->lookupTransform(
          image_rect_left->header.frame_id, base_frame_id_, 
          image_rect_left->header.stamp);
          
        sensor_data->local_transform.push_back(t.transform);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_INFO(
          node_->get_logger(), "Could not transform %s to %s: %s",
          image_rect_left->header.frame_id.c_str(), base_frame_id_.c_str(), ex.what());
        return;
      }
  } else {
    geometry_msgs::msg::Transform optTransform;
    rtabmap_conversions::transformToGeometryMsg(rtabmap::CameraModel::opticalRotation(), optTransform);
    sensor_data->local_transform.push_back(optTransform);
  }
  imagery_queue_.add(sensor_data);
}