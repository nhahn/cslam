#pragma once

#include "cslam/front_end/sensor_handler.h"

namespace cslam
{
    typedef message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image,
            sensor_msgs::msg::CameraInfo, sensor_msgs::msg::CameraInfo>
            ApproximateStereoSync;

    typedef message_filters::sync_policies::ExactTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image,
            sensor_msgs::msg::CameraInfo, sensor_msgs::msg::CameraInfo>
            ExactStereoSync;
    template <typename SyncPolicy>
    class StereoHandler : public SensorHandler
    {
    public:
        /**
         * @brief Initialization of parameters and ROS 2 objects
         *
         * @param node ROS 2 node handle
         */
        explicit StereoHandler(rclcpp::Node * node);
        ~StereoHandler(){};

        /**
         * @brief Callback receiving sync data from camera
         *
         * @param image_rect_left
         * @param image_rect_right
         * @param camera_info_left
         * @param camera_info_right
         * @param odom
         */
        void stereo_callback(
            const sensor_msgs::msg::Image::ConstSharedPtr image_rect_left,
            const sensor_msgs::msg::Image::ConstSharedPtr image_rect_right,
            const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_left,
            const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_right);


    private:

        image_transport::SubscriberFilter sub_image_rect_left_;
        image_transport::SubscriberFilter sub_image_rect_right_;
        image_transport::SubscriberFilter sub_image_global_;
        message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_camera_info_left_;
        message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_camera_info_right_;

        std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> stereo_synchronizer;
    };
} // namespace cslam