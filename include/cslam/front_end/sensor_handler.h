#pragma once

#include <rclcpp/rclcpp.hpp>
#include <atomic>
#include <rtabmap_msgs/msg/rgbd_image.hpp>
#include <rtabmap_msgs/msg/sensor_data.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <rtabmap/core/SensorData.h>
#include <sensor_msgs/msg/nav_sat_fix.hpp>

#include <rtabmap/utilite/UStl.h>
#include <mutex>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <message_filters/cache.h>

#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>

namespace cslam
{
    /**
     * @brief Interface class for handling sensor data
     * 
     */
    class SensorHandler
    {
    public:
        /**
         * @brief Virtual destructor
         */
        explicit SensorHandler(rclcpp::Node * node);
        ~SensorHandler(){};

        /**
         * @brief Process new data callback
         * 
         */
        typedef message_filters::sync_policies::ApproximateTime<
            rtabmap_msgs::msg::SensorData, nav_msgs::msg::Odometry>
            SensorSyncPolicy;
        std::unique_ptr<message_filters::Synchronizer<SensorSyncPolicy>> sensor_queue_;

        std::deque<std::pair<std::shared_ptr<rtabmap::SensorData>, nav_msgs::msg::Odometry::ConstSharedPtr>> process_queue_;
        bool enable_gps_recording_;
        std::string gps_topic_;
        sensor_msgs::msg::NavSatFix latest_gps_fix_;
        std::deque<sensor_msgs::msg::NavSatFix>
            received_gps_queue_;
        std::string base_frame_id_;
        std::atomic_ulong map_id{0}; int resetCounter = 4;
            
        protected:
            rclcpp::Node * node_;
            size_t max_queue_size_;
            std::shared_ptr<tf2_ros::Buffer>
                tf_buffer_;
            std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
            std::shared_ptr<rtabmap::StereoCameraModel> stereoCameraModel {nullptr};
        private:
            rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_subscriber_;
            rtabmap::Transform stereoTransform;
            bool alreadyRectified = true;
            std::shared_ptr<rtabmap::StereoCameraModel> fetchStereoModel(const rtabmap_msgs::msg::SensorData::ConstSharedPtr sensorMsg);
            message_filters::Subscriber<nav_msgs::msg::Odometry> sub_odometry_;


         void sensor_odom_callback(
                const rtabmap_msgs::msg::SensorData::ConstSharedPtr sensorMsg, 
                const nav_msgs::msg::Odometry::ConstSharedPtr odom);

                /**
             * @brief GPS data callback
             *
             * @param msg
             */
            void gps_callback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr msg);
    };
} // namespace cslam
