#pragma once
#include <chrono>

#include <rclcpp/rclcpp.hpp>

#include <rtabmap_msgs/msg/rgbd_image.hpp>
#include <rtabmap/core/Compression.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/RegistrationVis.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/VWDictionary.h>
#include <rtabmap/core/util2d.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/utilite/UStl.h>
#include <mutex>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/pass_through.h>

#include <message_filters/cache.h>

#include <std_msgs/msg/u_int32.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <cv_bridge/cv_bridge.hpp>

#include <chrono>
#include <cslam_common_interfaces/msg/keyframe_odom.hpp>
#include <cslam_common_interfaces/msg/keyframe_rgb.hpp>
#include <cslam_common_interfaces/msg/viz_point_cloud.hpp>
#include <cslam_common_interfaces/msg/inter_robot_loop_closure.hpp>
#include <cslam_common_interfaces/msg/inter_robot_matches.hpp>

#include <cslam_common_interfaces/msg/local_descriptors_request.hpp>
#include <cslam_common_interfaces/msg/local_image_descriptors.hpp>
#include <cslam_common_interfaces/msg/local_keyframe_match.hpp>
#include <cslam_common_interfaces/msg/intra_robot_loop_closure.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <deque>
#include <functional>
#include <nav_msgs/msg/odometry.hpp>
#include <thread>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <memory>

#include <rtabmap_conversions/MsgConversion.h>
#include "cslam/front_end/visualization_utils.h"
#include "lightglue_onnx/LightGlueOnnxRunner.hpp"
#include "lightglue_onnx/Configuration.hpp"
#include "cslam/front_end/utils/thread_pool.hpp"

#include "cslam/front_end/stereo_handler.h"
#include "cslam/front_end/rgbd_handler.h"
#include "cslam/front_end/utils/optical_flow.hpp"

namespace cslam {

/**
 * @brief Map management interface class 
 *
 */


/**
 * @brief Loop Closure Detection Management
 * - Receives keyframes from RTAB-map
 * - Generate keypoints from frames
 * - Sends/Receives keypoints from other robot frames
 * - Computes geometric verification
 *
 * @tparam DataHandlerType Depends on the type of input data (stereo, rgbd,
 * etc.)
 */
class MapManager:  public rclcpp::Node {
public:
  /**
   * @brief Initialization of parameters and ROS 2 objects
   *
   * @param node ROS 2 node handle
   */
  MapManager(rclcpp::NodeOptions ops);
  ~MapManager(){};

  /**
   * @brief Looks for loop closures in the current keyframe queue
   *
   */
  void process_new_sensor_data();

  /**
       * @brief Service callback to publish local descriptors
       *
       * @param request Image ID to send and matching info
       */
      void local_descriptors_request(
          cslam_common_interfaces::msg::LocalDescriptorsRequest::
              ConstSharedPtr request);

      /**
       * @brief Receives a local match and tries to compute a local loop closure
       *
       * @param msg
       */
      void receive_local_keyframe_match(
          cslam_common_interfaces::msg::LocalKeyframeMatch::ConstSharedPtr
              msg);

      /**
       * @brief Message callback to receive descriptors and compute
       *
       * @param msg local descriptors
       */
      void receive_local_image_descriptors(
          const std::shared_ptr<
              cslam_common_interfaces::msg::LocalImageDescriptors>
              msg);

      /**
       * @brief Computes local 3D descriptors from frame data and store them
       *
       * @param frame_data Full frame data
       */
      bool
      compute_local_descriptors(std::shared_ptr<rtabmap::SensorData> frame_data, const cv::Mat &img);

      /**
       * @brief converts descriptors to sensore data
       *
       * @param msg local descriptors
       * @return rtabmap::SensorData&
       */
      void local_descriptors_msg_to_sensor_data(
          const std::shared_ptr<
              cslam_common_interfaces::msg::LocalImageDescriptors>
              msg,
          rtabmap::SensorData &sensor_data);

      /**
       * @brief converts sensor data to descriptor msg
       *
       * @param sensor_data local descriptors
       * @param msg_data rtabmap_msgs::msg::RGBDImage&
       */
      void sensor_data_to_rgbd_msg(
          const std::shared_ptr<rtabmap::SensorData> sensor_data,
          rtabmap_msgs::msg::SensorData &msg_data,
          bool baselinkFrame = false);

      /**
       * @brief Generate a new keyframe according to the policy
       *
       * @param keyframe Sensor data
       * @return true A new keyframe is added to the map
       * @return false The frame is rejected
       */
      rtabmap::Transform compute_flow(const std::shared_ptr<rtabmap::SensorData> newData, const cv::Mat& toImg, rtabmap::RegistrationInfo &reg_info);

      /**
       * @brief Function to send the image to the python node
       *
       * @param keypoints_data keyframe keypoints data
       * @param gps_data GPS data
       */
      void send_keyframe(const std::pair<std::shared_ptr<rtabmap::SensorData>, std::shared_ptr<const nav_msgs::msg::Odometry>> &keypoints_data, const sensor_msgs::msg::NavSatFix * gps_data = nullptr);


      void send_visualization(const std::pair<std::shared_ptr<rtabmap::SensorData>, std::shared_ptr<const nav_msgs::msg::Odometry>> &keypoints_data);
      /**
       * @brief Send keypoints for visualizations
       *
       * @param keypoints_data keyframe keypoints data
       */
      void send_visualization_keypoints(const std::pair<std::shared_ptr<rtabmap::SensorData>, std::shared_ptr<const nav_msgs::msg::Odometry>> &keypoints_data);

      /**
       * @brief Send colored pointcloud for visualizations
       *
       * @param sensor_data RGBD image
       */
      void send_visualization_pointcloud(const std::shared_ptr<rtabmap::SensorData> &sensor_data);

      /**
       * @brief Clear images and large data fields in sensor data
       *
       * @param sensor_data frame data
       */
      void clear_sensor_data(std::shared_ptr<rtabmap::SensorData> sensor_data);

      void recover_odom_pose(cslam_common_interfaces::msg::InterRobotMatches::ConstSharedPtr match);
      /**
       * @brief Subsample pointcloud to reduce size for visualization
       * 
       * @param input_cloud 
       * @return pcl::PointCloud<pcl::PointXYZRGB>& 
       */
      sensor_msgs::msg::PointCloud2 visualization_pointcloud_voxel_subsampling(
                      const sensor_msgs::msg::PointCloud2 &input_cloud);

    protected:
        rclcpp::TimerBase::SharedPtr process_timer_;
        std::shared_ptr<rtabmap::SensorData> current_keyframe_;
        std::string sensor_type;
        std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

        rtabmap::Transform lastKFPose, currentPose;
        bool trackingLost = false;

        std::map<int, std::shared_ptr<rtabmap::SensorData>> local_descriptors_map_;

        unsigned int min_inliers_, max_nb_robots_, robot_id_, max_queue_size_,
            nb_local_keyframes_, map_manager_process_period_ms_;
        rclcpp::Subscription<cslam_common_interfaces::msg::InterRobotMatches>::SharedPtr recovery_subscriber_;
        rclcpp::Publisher<cslam_common_interfaces::msg::LocalKeyframeMatch>::SharedPtr add_recovered_publisher_;

        rclcpp::Subscription<
            cslam_common_interfaces::msg::LocalDescriptorsRequest>::SharedPtr
            send_local_descriptors_subscriber_;

        rclcpp::Publisher<
            cslam_common_interfaces::msg::LocalImageDescriptors>::SharedPtr
            local_descriptors_publisher_,
            visualization_local_descriptors_publisher_;

        rclcpp::Publisher<cslam_common_interfaces::msg::KeyframeRGB>::SharedPtr
            keyframe_data_publisher_;

        rclcpp::Publisher<cslam_common_interfaces::msg::KeyframeOdom>::SharedPtr
            keyframe_odom_publisher_;

        rclcpp::Publisher<cslam_common_interfaces::msg::VizPointCloud>::SharedPtr
            keyframe_pointcloud_publisher_;

        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
            keyframe_keypoint_viz_;

        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
            keyframe_matches_viz_;

        rclcpp::Subscription<
            cslam_common_interfaces::msg::LocalKeyframeMatch>::SharedPtr
            local_keyframe_match_subscriber_;

        rclcpp::Subscription<
            cslam_common_interfaces::msg::LocalImageDescriptors>::SharedPtr
            local_descriptors_subscriber_;

        std::shared_ptr<rtabmap::RegistrationVis> inter_registration_;
        std::shared_ptr<rtabmap::RegistrationVis> intra_registration_;
        std::shared_ptr<rtabmap::RegistrationVis> f2f_registration_;

        rclcpp::Publisher<
            cslam_common_interfaces::msg::InterRobotLoopClosure>::SharedPtr
            inter_robot_loop_closure_publisher_;

        rclcpp::Publisher<
            cslam_common_interfaces::msg::IntraRobotLoopClosure>::SharedPtr
            intra_robot_loop_closure_publisher_;

        rclcpp::Publisher<
            diagnostic_msgs::msg::KeyValue>::SharedPtr
            log_publisher_;
        unsigned int log_total_local_descriptors_cumulative_communication_;
        bool enable_logs_, external_odom_;

        float keyframe_generation_ratio_threshold_;
        int min_3d_keypoints_;
        unsigned long currentMapId_ = 0;
        unsigned int visualization_period_ms_;
        bool enable_visualization_;
        float visualization_voxel_size_, visualization_max_range_;

        std::string global_image_topic_;
        tf2::Transform base_transform_; bool hasTransform_;
        rtabmap::Feature2D * detector_;
        std::shared_ptr<lightglue::LightGlueOnnxRunner> lightglueMatcher;
        lightglue::Configuration lightglueConfig;
        std::shared_ptr<OpticalFlow> optical_matcher;
    private:
        cv::Mat keypointViz, matchesViz;


        nav_msgs::msg::Odometry calcOdom;
        geometry_msgs::msg::TransformStamped odomTf;
        
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
        std::shared_ptr<SensorHandler> sensor_handler_ {nullptr};
        rclcpp::CallbackGroup::SharedPtr sensorDataCB;
        bool setMatches(rtabmap::Signature &from, rtabmap::Signature &to);
        std::pair<std::shared_ptr<rtabmap::Signature>, std::shared_ptr<rtabmap::Signature>> computeMatches(const rtabmap::SensorData& from, const rtabmap::SensorData& to);
        rtabmap::ParametersMap rtabmap_parameters;
        sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg_;
        std::mutex map_mutex, prev_frame_mutex, current_pose_mutex;
        ThreadPool workerPool;
        //ThreadPool keypointExtractorPool, matcherPool, poseEstimatorPool;
  

};


} 