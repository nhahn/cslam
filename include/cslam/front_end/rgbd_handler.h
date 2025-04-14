#pragma once

#include "cslam/front_end/sensor_handler.h"

namespace cslam
{
    typedef message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image,
            sensor_msgs::msg::CameraInfo>
            ApproximateRGBDSync;

    typedef message_filters::sync_policies::ExactTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image,
            sensor_msgs::msg::CameraInfo>
            ExactRGBDSync;
    template <typename SyncPolicy>
    class RGBDHandler : public SensorHandler
    {
    public:
        /**
         * @brief Initialization of parameters and ROS 2 objects
         *
         * @param node ROS 2 node handle
         */
        explicit RGBDHandler(rclcpp::Node * node);
        ~RGBDHandler(){};


      /**
       * @brief Callback receiving sync data from camera
       *
       * @param image_rgb
       * @param image_depth
       * @param camera_info_rgb
       * @param camera_info_depth
       * @param odom
       */
      void rgbd_callback(
          const sensor_msgs::msg::Image::ConstSharedPtr image_rect_rgb,
          const sensor_msgs::msg::Image::ConstSharedPtr image_rect_depth,
          const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_rgb);


    private:
        image_transport::SubscriberFilter sub_image_color_;
        message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_camera_info_color_;
        image_transport::SubscriberFilter sub_image_depth_;
        message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_camera_info_depth_;

        std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> rgbd_synchronizer;

    };
} // namespace cslam

