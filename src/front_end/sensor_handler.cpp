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
  node->declare_parameter<std::string>("frontend.odom_topic", "odom");
  node->get_parameter("frontend.use_external_odom", external_odom_);
  node->declare_parameter<float>("frontend.odom_sync_period", 0.1);
  defaultCameraTransform = rtabmap::CameraModel::opticalRotation();
  rtabmap_conversions::transformToGeometryMsg(defaultCameraTransform, rosCameraTransform);

  if (external_odom_) {
    sub_odometry_.subscribe(node,
                            node->get_parameter("frontend.odom_topic").as_string(),
                            qos);


      sensor_synchronizer_ = std::make_unique<message_filters::Synchronizer<SensorSyncPolicy>>(
          SensorSyncPolicy(max_queue_size_), imagery_queue_, sub_odometry_);
      sensor_synchronizer_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(node_->get_parameter("frontend.odom_sync_period").as_double()));
      sensor_synchronizer_->registerCallback(std::bind(&SensorHandler::sensor_odom_callback, this, std::placeholders::_1,
          std::placeholders::_2));
  } else {
    imagery_queue_.registerCallback([this](const rtabmap_msgs::msg::SensorData::ConstSharedPtr sensorMsg) {
      this->sensor_odom_callback(sensorMsg, nullptr);
    });
  }

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

std::shared_ptr<rtabmap::StereoCameraModel> SensorHandler::fetchStereoModel(const rtabmap_msgs::msg::SensorData::ConstSharedPtr sensorMsg) {
  if (!stereoCameraModel) {
      rtabmap::Transform stereoTransform;
      rtabmap::Transform localTransform = rtabmap_conversions::transformFromGeometryMsg(sensorMsg->local_transform[0]);
      if (!alreadyRectified) {
        stereoTransform = rtabmap_conversions::getTransform(
            sensorMsg->right_camera_info[0].header.frame_id, sensorMsg->left_camera_info[0].header.frame_id,
            sensorMsg->left_camera_info[0].header.stamp, *tf_buffer_, 0.1);
        if (stereoTransform.isNull()) {
          RCLCPP_ERROR(node_->get_logger(),
                      "Already rectified false but we cannot get TF between the "
                      "two cameras! (between frames %s and %s)",
                      sensorMsg->right_camera_info[0].header.frame_id.c_str(),
                      sensorMsg->left_camera_info[0].header.frame_id.c_str());
          return nullptr;
        } else if (stereoTransform.isIdentity()) {
          RCLCPP_ERROR(node_->get_logger(),
                      "Already rectified false but we cannot get a valid TF "
                      "between the two cameras! "
                      "Identity transform returned between left and right "
                      "cameras. Verify that if TF between "
                      "the cameras is valid: \"rosrun tf tf_echo %s %s\".",
                      sensorMsg->right_camera_info[0].header.frame_id.c_str(),
                      sensorMsg->left_camera_info[0].header.frame_id.c_str());
          return nullptr;
        }
      }
      stereoCameraModel = std::make_shared<rtabmap::StereoCameraModel>(rtabmap_conversions::stereoCameraModelFromROS(sensorMsg->left_camera_info[0], sensorMsg->right_camera_info[0],
                                                localTransform, stereoTransform));
      if (stereoCameraModel->baseline() == 0 && alreadyRectified) {
        stereoTransform = rtabmap_conversions::getTransform(
            sensorMsg->left_camera_info[0].header.frame_id, sensorMsg->right_camera_info[0].header.frame_id,
            sensorMsg->left_camera_info[0].header.stamp, *tf_buffer_, 0.1);

        if (!stereoTransform.isNull() && stereoTransform.x() > 0) {
          static bool warned = false;
          if (!warned) {
            RCLCPP_WARN(
                node_->get_logger(),
                "Right camera info doesn't have Tx set but we are assuming that "
                "stereo images are already rectified. While "
                "not "
                "recommended, we used TF to get the baseline (%s->%s = %fm) for "
                "convenience (e.g., D400 ir stereo issue). It is preferred to "
                "feed "
                "a valid right camera info if stereo images are already "
                "rectified. This message is only printed once...",
                sensorMsg->right_camera_info[0].header.frame_id.c_str(),
                sensorMsg->left_camera_info[0].header.frame_id.c_str(), stereoTransform.x());
            warned = true;
          }
          stereoCameraModel = std::make_shared<rtabmap::StereoCameraModel>(
              stereoCameraModel->left().fx(), stereoCameraModel->left().fy(),
              stereoCameraModel->left().cx(), stereoCameraModel->left().cy(),
              stereoTransform.x(), stereoCameraModel->localTransform(),
              stereoCameraModel->left().imageSize());
        }
      }

      if (alreadyRectified && stereoCameraModel->baseline() <= 0) {
        RCLCPP_ERROR(
            node_->get_logger(),
            "The stereo baseline (%f) should be positive (baseline=-Tx/fx). We "
            "assume a horizontal left/right stereo "
            "setup where the Tx (or P(0,3)) is negative in the right camera info "
            "msg.",
            stereoCameraModel->baseline());
        return nullptr;
      }

      if (stereoCameraModel->baseline() > 10.0) {
        static bool shown = false;
        if (!shown) {
          RCLCPP_WARN(
              node_->get_logger(),
              "Detected baseline (%f m) is quite large! Is your "
              "right camera_info P(0,3) correctly set? Note that "
              "baseline=-P(0,3)/P(0,0). This warning is printed only once.",
              stereoCameraModel->baseline());
          shown = true;
        }
      }
      RCLCPP_INFO(node_->get_logger(), "Stereo cam setup: %f -- %f %f %f %f %f", stereoCameraModel->baseline(), stereoCameraModel->left().fx(), stereoCameraModel->left().fy(),
              stereoCameraModel->left().cx(), stereoCameraModel->left().cy(),
              stereoTransform.x());
      RCLCPP_INFO(node_->get_logger(), "TF for cameras: %s", stereoCameraModel->localTransform().prettyPrint().c_str());
    }
    return stereoCameraModel;
}

bool SensorHandler::hasData() {
  std::lock_guard<std::mutex> queue_lock(queueMutex);
  return !process_queue_.empty();
}

std::pair<std::shared_ptr<rtabmap::SensorData>, nav_msgs::msg::Odometry::ConstSharedPtr> SensorHandler::getNextPair() {
  std::lock_guard<std::mutex> queue_lock(queueMutex);
  if (!process_queue_.empty())
  {
    auto pair = process_queue_.front();
    process_queue_.pop_front();
    return pair;
  }
  return std::make_pair(nullptr, nullptr);
}


 void SensorHandler::sensor_odom_callback(
        const rtabmap_msgs::msg::SensorData::ConstSharedPtr sensorMsg, 
        const nav_msgs::msg::Odometry::ConstSharedPtr odom){
          // If odom tracking failed, do not process the frame
    if (odom && odom->pose.covariance[0] > 1000)
    {
      RCLCPP_WARN(node_->get_logger(), "Odom tracking failed, skipping frame");
      // if (odom->pose.covariance[0] > 9000 && process_queue_.size()) { //We've lost tracking -- reset the pose graph
      //   if (resetCounter-- == 0) {
      //     RCLCPP_WARN(node_->get_logger(), "Odom tracking failed, resetting...");
      //     process_queue_.clear();
      //     map_id++;
      //   }
      // }
      return;
    } 

    auto sensorData = std::make_shared<rtabmap::SensorData>(rtabmap_conversions::sensorDataFromROS(*sensorMsg));
    sensor_frame = sensorMsg->header.frame_id;
    if (sensorData->stereoCameraModels().size() > 0) {
      auto model = fetchStereoModel(sensorMsg);
      if (!model) return;
      sensorData->setStereoCameraModel(*model);
    }

    if (sensorData->isValid()) {
      std::lock_guard<std::mutex> queue_lock(queueMutex);
      process_queue_.push_back(std::make_pair(sensorData, odom));
      if (process_queue_.size() > max_queue_size_)
      {
        // Remove the oldest keyframes if we exceed the maximum size
        process_queue_.pop_front();
        // RCLCPP_DEBUG(
        //     node_->get_logger(),
        //     "SensorHandler: Maximum queue size (%d) exceeded, the oldest element was removed.",
        //     max_queue_size_);
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