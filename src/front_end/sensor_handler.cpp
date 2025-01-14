#include "cslam/front_end/sensor_handler.h"
#include <functional>
#include <rtabmap_conversions/MsgConversion.h>

using namespace rtabmap;
using namespace cslam;

SensorHandler::SensorHandler(rclcpp::Node * node) : node_(node) {
  auto qos = rclcpp::SensorDataQoS().get_rmw_qos_profile();
  node->get_parameter("frontend.max_queue_size", max_queue_size_);
  node->get_parameter("evaluation.enable_gps_recording",
                        enable_gps_recording_);
  node->get_parameter("evaluation.gps_topic",
                        gps_topic_);
  node->get_parameter("frontend.sensor_base_frame_id", base_frame_id_);
  node_->declare_parameter<float>("frontend.sync_period", 0.2);

  sub_odometry_.subscribe(node,
                          node->get_parameter("frontend.odom_topic").as_string(),
                          qos);


    message_filters::NullFilter<rtabmap_msgs::msg::SensorData> f0;
    sensor_queue_ = std::make_unique<message_filters::Synchronizer<SensorSyncPolicy>>(
        SensorSyncPolicy(max_queue_size_), f0, sub_odometry_);
    sensor_queue_->registerCallback(std::bind(&SensorHandler::sensor_odom_callback, this, std::placeholders::_1,
        std::placeholders::_2));
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
      
    if (enable_gps_recording_)
    {
      gps_subscriber_ = node_->create_subscription<sensor_msgs::msg::NavSatFix>(
          gps_topic_, 100,
          std::bind(&SensorHandler::gps_callback, this,
                    std::placeholders::_1));
    }

  }

  void SensorHandler::gps_callback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr msg)
  {
    latest_gps_fix_ = *msg;
  }


 void SensorHandler::sensor_odom_callback(
        const rtabmap_msgs::msg::SensorData::ConstSharedPtr sensorMsg, 
        const nav_msgs::msg::Odometry::ConstSharedPtr odom){
          // If odom tracking failed, do not process the frame
    if (odom->pose.covariance[0] > 1000)
    {
      RCLCPP_WARN(node_->get_logger(), "Odom tracking failed, skipping frame");
      return;
    } 

    auto sensorData = std::make_shared<rtabmap::SensorData>(rtabmap_conversions::sensorDataFromROS(*sensorMsg));
    if (sensorData->isValid()) {

        process_queue_.push_back(std::make_pair(sensorData, odom));
        if (process_queue_.size() > max_queue_size_)
        {
          // Remove the oldest keyframes if we exceed the maximum size
          process_queue_.pop_front();
          RCLCPP_DEBUG(
              node_->get_logger(),
              "RGBD: Maximum queue size (%d) exceeded, the oldest element was removed.",
              max_queue_size_);
        }

        if (enable_gps_recording_) {
          received_gps_queue_.push_back(latest_gps_fix_);
          if (received_gps_queue_.size() > max_queue_size_)
          {
            received_gps_queue_.pop_front();
          }
        }
    }
    
}