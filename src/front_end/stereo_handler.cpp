#include "cslam/front_end/stereo_handler.h"
#include <rtabmap_conversions/MsgConversion.h>
#include "tf2_eigen/tf2_eigen.hpp"

using namespace rtabmap;
using namespace cslam;

StereoHandler::StereoHandler(rclcpp::Node * node)
    : RGBDHandler(node) {
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

  stereo_synchronizer = std::make_unique<message_filters::Synchronizer<StereoPolicy>>(
        StereoPolicy(max_queue_size_), sub_image_rect_left_, sub_image_rect_right_,
        sub_camera_info_left_, sub_camera_info_right_);
  stereo_synchronizer->registerCallback(
        std::bind(&StereoHandler::stereo_callback, this, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3,
                    std::placeholders::_4));

    }

void StereoHandler::stereo_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr image_rect_left,
    const sensor_msgs::msg::Image::ConstSharedPtr image_rect_right,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_left,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_right) {

  if (!(image_rect_left->encoding.compare(
            sensor_msgs::image_encodings::TYPE_8UC1) == 0 ||
        image_rect_left->encoding.compare(sensor_msgs::image_encodings::MONO8) ==
            0 ||
        image_rect_left->encoding.compare(sensor_msgs::image_encodings::MONO16) ==
            0 ||
        image_rect_left->encoding.compare(sensor_msgs::image_encodings::BGR8) ==
            0 ||
        image_rect_left->encoding.compare(sensor_msgs::image_encodings::RGB8) ==
            0 ||
        image_rect_left->encoding.compare(sensor_msgs::image_encodings::BGRA8) ==
            0 ||
        image_rect_left->encoding.compare(sensor_msgs::image_encodings::RGBA8) ==
            0) ||
      !(image_rect_right->encoding.compare(
            sensor_msgs::image_encodings::TYPE_8UC1) == 0 ||
        image_rect_right->encoding.compare(sensor_msgs::image_encodings::MONO8) ==
            0 ||
        image_rect_right->encoding.compare(
            sensor_msgs::image_encodings::MONO16) == 0 ||
        image_rect_right->encoding.compare(sensor_msgs::image_encodings::BGR8) ==
            0 ||
        image_rect_right->encoding.compare(sensor_msgs::image_encodings::RGB8) ==
            0 ||
        image_rect_right->encoding.compare(sensor_msgs::image_encodings::BGRA8) ==
            0 ||
        image_rect_right->encoding.compare(sensor_msgs::image_encodings::RGBA8) ==
            0)) {
    RCLCPP_ERROR(
        node_->get_logger(),
        "Input type must be image=mono8,mono16,rgb8,bgr8,rgba8,bgra8 (mono8 "
        "recommended), received types are %s (left) and %s (right)",
        image_rect_left->encoding.c_str(), image_rect_right->encoding.c_str());
    return;
  }
  rclcpp::Time stamp = image_rect_left->header.stamp;

  if (image_rect_left->data.size() && image_rect_right->data.size()) {
    bool alreadyRectified = true;

    if (!stereoCameraModel) {

      rtabmap::Transform stereoTransform;
      if (!alreadyRectified) {
        stereoTransform = rtabmap_conversions::getTransform(
            camera_info_right->header.frame_id, camera_info_left->header.frame_id,
            camera_info_left->header.stamp, *tf_buffer_, 0.1);
        if (stereoTransform.isNull()) {
          RCLCPP_ERROR(node_->get_logger(),
                      "Parameter %s is false but we cannot get TF between the "
                      "two cameras! (between frames %s and %s)",
                      Parameters::kRtabmapImagesAlreadyRectified().c_str(),
                      camera_info_right->header.frame_id.c_str(),
                      camera_info_left->header.frame_id.c_str());
          return;
        } else if (stereoTransform.isIdentity()) {
          RCLCPP_ERROR(node_->get_logger(),
                      "Parameter %s is false but we cannot get a valid TF "
                      "between the two cameras! "
                      "Identity transform returned between left and right "
                      "cameras. Verify that if TF between "
                      "the cameras is valid: \"rosrun tf tf_echo %s %s\".",
                      Parameters::kRtabmapImagesAlreadyRectified().c_str(),
                      camera_info_right->header.frame_id.c_str(),
                      camera_info_left->header.frame_id.c_str());
          return;
        }
      }
      stereoCameraModel = std::make_shared<rtabmap::StereoCameraModel>(rtabmap_conversions::stereoCameraModelFromROS(*camera_info_left, *camera_info_right,
                                                Transform::getIdentity(), stereoTransform));
      if (stereoCameraModel->baseline() == 0 && alreadyRectified) {
        stereoTransform = rtabmap_conversions::getTransform(
            camera_info_left->header.frame_id, camera_info_right->header.frame_id,
            camera_info_left->header.stamp, *tf_buffer_, 0.1);

        if (!stereoTransform.isNull() && stereoTransform.x() > 0) {
          static bool warned = false;
          if (!warned) {
            RCLCPP_WARN(
                node_->get_logger(),
                "Right camera info doesn't have Tx set but we are assuming that "
                "stereo images are already rectified (see %s parameter). While "
                "not "
                "recommended, we used TF to get the baseline (%s->%s = %fm) for "
                "convenience (e.g., D400 ir stereo issue). It is preferred to "
                "feed "
                "a valid right camera info if stereo images are already "
                "rectified. This message is only printed once...",
                rtabmap::Parameters::kRtabmapImagesAlreadyRectified().c_str(),
                camera_info_right->header.frame_id.c_str(),
                camera_info_left->header.frame_id.c_str(), stereoTransform.x());
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
        return;
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

    //TODO for now we're testing to see if all mono images are better for place recognition
    auto ptrImageLeft = cv_bridge::toCvCopy(
        image_rect_left, image_rect_left->encoding.compare(
                           sensor_msgs::image_encodings::TYPE_8UC1) == 0 ||
                               image_rect_left->encoding.compare(
                                   sensor_msgs::image_encodings::MONO8) == 0
                           ? ""
                      //  : image_rect_left->encoding.compare(
                      //        sensor_msgs::image_encodings::MONO16) != 0
                      //      ? "bgr8"
                           : "mono8");
    auto ptrImageRight = cv_bridge::toCvCopy(
        image_rect_right, image_rect_right->encoding.compare(
                            sensor_msgs::image_encodings::TYPE_8UC1) == 0 ||
                                image_rect_right->encoding.compare(
                                    sensor_msgs::image_encodings::MONO8) == 0
                            ? ""
                            : "mono8");

    auto data = std::make_shared<rtabmap::SensorData>(
        ptrImageLeft->image, ptrImageRight->image, *stereoCameraModel.get(),
        0, rtabmap_conversions::timestampFromROS(stamp));


    received_imagery_queue_.push_back(data);
    if (received_imagery_queue_.size() > max_queue_size_) {
      // Remove the oldest keyframes if we exceed the maximum size
      received_imagery_queue_.pop_front();
    }

    if (enable_gps_recording_) {
        received_gps_queue_.push_back(latest_gps_fix_);
        if (received_gps_queue_.size() > max_queue_size_)
        {
        received_gps_queue_.pop_front();
        }
    }
  } else {
    RCLCPP_WARN(node_->get_logger(), "Odom: input images empty?!");
  }
}

void StereoHandler::send_visualization(const std::pair<std::shared_ptr<rtabmap::SensorData>, std::shared_ptr<const nav_msgs::msg::Odometry>> &keypoints_data)
{
  send_visualization_keypoints(keypoints_data);
}
