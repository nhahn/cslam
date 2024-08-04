#ifndef _STEREOHANDLER_H_
#define _STEREOHANDLER_H_

#include "cslam/front_end/rgbd_handler.h"

namespace cslam
{

    class StereoHandler : public RGBDHandler
    {
    public:
        /**
         * @brief Initialization of parameters and ROS 2 objects
         *
         * @param node ROS 2 node handle
         */
        StereoHandler(rclcpp::Node * node);
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

        /**
         * @brief Send keypoints for visualizations
         * 
         * @param keypoints_data keyframe keypoints data
         */
        virtual void send_visualization(const std::pair<std::shared_ptr<rtabmap::SensorData>, std::shared_ptr<const nav_msgs::msg::Odometry>> &keypoints_data);

    private:
        std::shared_ptr<rtabmap::StereoCameraModel> stereoCameraModel {nullptr};

        image_transport::SubscriberFilter sub_image_rect_left_;
        image_transport::SubscriberFilter sub_image_rect_right_;
        message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_camera_info_left_;
        message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_camera_info_right_;
        typedef message_filters::sync_policies::ExactTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image,
            sensor_msgs::msg::CameraInfo, sensor_msgs::msg::CameraInfo>
            StereoPolicy;
        std::unique_ptr<message_filters::Synchronizer<StereoPolicy>> stereo_synchronizer;
    };
} // namespace cslam
#endif