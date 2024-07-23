#include "cslam/front_end/rgbd_handler.h"
#include <rtabmap_conversions/MsgConversion.h>
#include <filesystem>
#include <tuple>
// For visualization
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/core/eigen.hpp>
#include "tf2_eigen/tf2_eigen.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace rtabmap;
using namespace cslam;

#define MAP_FRAME_ID(id) "robot" + std::to_string(id) + "_map"

RGBDHandler::RGBDHandler(rclcpp::Node * node)
    : node_(node)
{
  node_->declare_parameter<std::string>("frontend.color_image_topic", "color/image");
  node_->declare_parameter<std::string>("frontend.depth_image_topic", "depth/image");
  node_->declare_parameter<std::string>("frontend.color_camera_info_topic",
                                        "color/camera_info");
  node_->declare_parameter<std::string>("frontend.superpoint_model", "/models/superpoint_1536.onnx");
  node_->declare_parameter<std::string>("frontend.lightglue_model", "/models/superpoint_lightglue_fused_fp16.onnx");
  node_->declare_parameter<std::string>("frontend.odom_topic", "odom");
  node_->declare_parameter<float>("frontend.keyframe_generation_ratio_threshold", 0.0);
  node_->declare_parameter<std::string>("frontend.sensor_base_frame_id", ""); // If empty we assume that the camera link is the base link
  node_->declare_parameter<bool>("evaluation.enable_logs", false);
  node_->get_parameter("frontend.max_queue_size", max_queue_size_);
  node_->get_parameter("frontend.keyframe_generation_ratio_threshold", keyframe_generation_ratio_threshold_);
  node_->get_parameter("frontend.sensor_base_frame_id", base_frame_id_);
  node_->get_parameter("visualization.enable",
                       enable_visualization_);
  node_->get_parameter("visualization.publishing_period_ms",
                       visualization_period_ms_);
  node_->declare_parameter("frontend.matcher_threshold", 0.1f);
  node_->get_parameter("visualization.voxel_size",
                       visualization_voxel_size_);
  node_->get_parameter("visualization.max_range",
                       visualization_max_range_);

  node_->get_parameter("evaluation.enable_logs",
                       enable_logs_);
  node_->get_parameter("evaluation.enable_gps_recording",
                      enable_gps_recording_);
  node_->get_parameter("evaluation.gps_topic",
                      gps_topic_);

  node_->declare_parameter<std::string>("frontend.global_descriptor_image_topic", "");
  global_image_topic_ = node_->get_parameter("frontend.global_descriptor_image_topic").as_string();

  //Fetch all the rtabmap parameters and then assign them to an rtabmap param setup
  //Initialize the interface, or we get an error
  std::map<std::string, rclcpp::Parameter> pmap;
  node_->get_parameters("rtabmap", pmap);
  auto paramList = rtabmap::Parameters::getDefaultParameters();
  for (auto const& x : paramList) {
    node_->declare_parameter<std::string>("rtabmap." + x.first, x.second);
    auto val = node_->get_parameter("rtabmap." + x.first);
    rtabmap_parameters.insert_or_assign(x.first, val.as_string());
  }
  
  detector_ = rtabmap::Feature2D::create(rtabmap_parameters);
  detector_->parseParameters(rtabmap_parameters);
  lightglueConfig = lightglue::Configuration{
    node_->get_parameter("frontend.superpoint_model").as_string(),
    node_->get_parameter("frontend.lightglue_model").as_string()
  };
  lightglueConfig.grayScale = true;
  lightglueConfig.matcherUseTrt = false;
  lightglueConfig.extractorUseTrt = true;
  lightglueMatcher = std::make_shared<lightglue::LightGlueDecoupleOnnxRunner>();
  lightglueMatcher->InitOrtEnv(lightglueConfig);
  lightglueMatcher->SetMatchThresh(node_->get_parameter("frontend.matcher_threshold").as_double());

  if (keyframe_generation_ratio_threshold_ > 0.99)
  {
    generate_new_keyframes_based_on_inliers_ratio_ = false;
  }
  else
  {
    generate_new_keyframes_based_on_inliers_ratio_ = true;
  }

  nb_local_keyframes_ = 0;

  sub_odometry_.subscribe(node_,
                          node_->get_parameter("frontend.odom_topic").as_string(),
                          rclcpp::QoS(max_queue_size_)
                              .reliability((rmw_qos_reliability_policy_t)2)
                              .get_rmw_qos_profile());

  // Service to extract and publish local image descriptors to another robot
  send_local_descriptors_subscriber_ = node_->create_subscription<
      cslam_common_interfaces::msg::LocalDescriptorsRequest>(
      "cslam/local_descriptors_request", 100,
      std::bind(&RGBDHandler::local_descriptors_request, this,
                std::placeholders::_1));

  // Parameters
  node_->get_parameter("frontend.max_queue_size", max_queue_size_);
  node_->get_parameter("frontend.inter_pnp_min_inliers", min_inliers_);
  node_->get_parameter("max_nb_robots", max_nb_robots_);
  node_->get_parameter("robot_id", robot_id_);

  // Publisher for global descriptors
  keyframe_data_publisher_ =
      node_->create_publisher<cslam_common_interfaces::msg::KeyframeRGB>(
          "cslam/keyframe_data", 100);

  // Publisher for odometry with ID
  keyframe_odom_publisher_ =
      node_->create_publisher<cslam_common_interfaces::msg::KeyframeOdom>(
          "cslam/keyframe_odom", 100);

  rclcpp::SubscriptionOptions intraOptions;
  intraOptions.callback_group = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  // Local matches subscription
  local_keyframe_match_subscriber_ = node_->create_subscription<
      cslam_common_interfaces::msg::LocalKeyframeMatch>(
      "cslam/local_keyframe_match", 100,
      std::bind(&RGBDHandler::receive_local_keyframe_match, this,
                std::placeholders::_1));

  // Publishers to other robots local descriptors subscribers
  std::string local_descriptors_topic = "/cslam/local_descriptors";
  local_descriptors_publisher_ = node_->create_publisher<
      cslam_common_interfaces::msg::LocalImageDescriptors>(local_descriptors_topic, 100);

  if (enable_visualization_)
  {
    visualization_local_descriptors_publisher_ = node_->create_publisher<
        cslam_common_interfaces::msg::LocalImageDescriptors>("/cslam/viz/local_descriptors", 100);

    keyframe_pointcloud_publisher_ = node_->create_publisher<cslam_common_interfaces::msg::VizPointCloud>(
        "/cslam/viz/keyframe_pointcloud", 100);
  }

  rclcpp::SubscriptionOptions interOptions;
  interOptions.callback_group = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Subscriber for local descriptors
  local_descriptors_subscriber_ = node_->create_subscription<
      cslam_common_interfaces::msg::LocalImageDescriptors>(
      "/cslam/local_descriptors", 100,
      std::bind(&RGBDHandler::receive_local_image_descriptors, this,
                std::placeholders::_1));

  // Registration settings
  auto interParams = rtabmap::ParametersMap(rtabmap_parameters);
  interParams.insert_or_assign(rtabmap::Parameters::kVisMinInliers(), std::to_string(node_->get_parameter("frontend.inter_pnp_min_inliers").as_int()));
  inter_registration_.parseParameters(interParams);

  auto intraParams = rtabmap::ParametersMap(rtabmap_parameters);
  intraParams.insert_or_assign(rtabmap::Parameters::kVisMinInliers(), std::to_string(node_->get_parameter("frontend.intra_pnp_min_inliers").as_int()));
  intra_registration_.parseParameters(intraParams);

  // Intra-robot loop closure publisher
  intra_robot_loop_closure_publisher_ = node_->create_publisher<
      cslam_common_interfaces::msg::IntraRobotLoopClosure>(
      "cslam/intra_robot_loop_closure", 100);

  // Publisher for inter robot loop closure to all robots
  inter_robot_loop_closure_publisher_ = node_->create_publisher<
      cslam_common_interfaces::msg::InterRobotLoopClosure>(
      "/cslam/inter_robot_loop_closure", 100);

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Subscriber for RGBD images
  sub_image_color_.subscribe(
      node_, node_->get_parameter("frontend.color_image_topic").as_string(), "raw",
      rclcpp::QoS(max_queue_size_)
          .reliability((rmw_qos_reliability_policy_t)2)
          .get_rmw_qos_profile());
  sub_image_depth_.subscribe(
      node_, node_->get_parameter("frontend.depth_image_topic").as_string(), "raw",
      rclcpp::QoS(max_queue_size_)
          .reliability((rmw_qos_reliability_policy_t)2)
          .get_rmw_qos_profile());
  sub_camera_info_color_.subscribe(
      node_, node_->get_parameter("frontend.color_camera_info_topic").as_string(),
      rclcpp::QoS(max_queue_size_)
          .reliability((rmw_qos_reliability_policy_t)2)
          .get_rmw_qos_profile());

  rgbd_sync_policy_ = new message_filters::Synchronizer<RGBDSyncPolicy>(
      RGBDSyncPolicy(max_queue_size_), sub_image_color_, sub_image_depth_,
      sub_camera_info_color_, sub_odometry_);
  rgbd_sync_policy_->registerCallback(
      std::bind(&RGBDHandler::rgbd_callback, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3,
                std::placeholders::_4));

  if (enable_gps_recording_)
  {
    gps_subscriber_ = node_->create_subscription<sensor_msgs::msg::NavSatFix>(
        gps_topic_, 100,
        std::bind(&RGBDHandler::gps_callback, this,
                  std::placeholders::_1));
  }

  if (enable_logs_){
    log_total_local_descriptors_cumulative_communication_ = 0;
    log_publisher_ = node_->create_publisher<diagnostic_msgs::msg::KeyValue>(
        "cslam/log_info", 100);
  }
}

void RGBDHandler::rgbd_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr image_rect_rgb,
    const sensor_msgs::msg::Image::ConstSharedPtr image_rect_depth,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_rgb,
    const nav_msgs::msg::Odometry::ConstSharedPtr odom)
{
  // If odom tracking failed, do not process the frame
  if (odom->pose.covariance[0] > 1000)
  {
    RCLCPP_WARN(node_->get_logger(), "Odom tracking failed, skipping frame");
    return;
  }

  if (!(image_rect_rgb->encoding.compare(sensor_msgs::image_encodings::TYPE_8UC1) == 0 ||
        image_rect_rgb->encoding.compare(sensor_msgs::image_encodings::MONO8) == 0 ||
        image_rect_rgb->encoding.compare(sensor_msgs::image_encodings::MONO16) == 0 ||
        image_rect_rgb->encoding.compare(sensor_msgs::image_encodings::BGR8) == 0 ||
        image_rect_rgb->encoding.compare(sensor_msgs::image_encodings::RGB8) == 0 ||
        image_rect_rgb->encoding.compare(sensor_msgs::image_encodings::BGRA8) == 0 ||
        image_rect_rgb->encoding.compare(sensor_msgs::image_encodings::RGBA8) == 0 ||
        image_rect_rgb->encoding.compare(sensor_msgs::image_encodings::BAYER_GRBG8) == 0) ||
      !(image_rect_depth->encoding.compare(sensor_msgs::image_encodings::TYPE_16UC1) == 0 ||
        image_rect_depth->encoding.compare(sensor_msgs::image_encodings::TYPE_32FC1) == 0 ||
        image_rect_depth->encoding.compare(sensor_msgs::image_encodings::MONO16) == 0))
  {
    RCLCPP_ERROR(node_->get_logger(), "Input type must be image=mono8,mono16,rgb8,bgr8,bgra8,rgba8 and "
                                      "image_depth=32FC1,16UC1,mono16. Current rgb=%s and depth=%s",
                 image_rect_rgb->encoding.c_str(),
                 image_rect_depth->encoding.c_str());
    return;
  }

  rclcpp::Time stamp = rtabmap_conversions::timestampFromROS(image_rect_rgb->header.stamp) > rtabmap_conversions::timestampFromROS(image_rect_depth->header.stamp) ? image_rect_rgb->header.stamp : image_rect_depth->header.stamp;

  cv_bridge::CvImageConstPtr ptr_image = cv_bridge::toCvShare(image_rect_rgb);
  cv_bridge::CvImageConstPtr ptr_depth = cv_bridge::toCvShare(image_rect_depth);

  CameraModel camera_model = rtabmap_conversions::cameraModelFromROS(*camera_info_rgb);

  auto data = std::make_shared<rtabmap::SensorData>(
      ptr_image->image, ptr_depth->image,
      camera_model,
      0,
      rtabmap_conversions::timestampFromROS(stamp));
  

  received_data_queue_.push_back(std::make_pair(data, odom));
  if (received_data_queue_.size() > max_queue_size_)
  {
    // Remove the oldest keyframes if we exceed the maximum size
    received_data_queue_.pop_front();
    RCLCPP_WARN(
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

std::vector<cv::Point2f> NormKeypoints(std::vector<cv::KeyPoint> kpts, int h, int w)
{
    cv::Size size(w, h);
    cv::Point2f shift(static_cast<float>(w) / 2, static_cast<float>(h) / 2);
    float scale = static_cast<float>((std::max)(w, h)) / 2;

    std::vector<cv::Point2f> normalizedKpts;
    normalizedKpts.reserve(kpts.size());
    for (const cv::KeyPoint &kpt : kpts)
    {
        cv::Point2f normalizedKpt = (kpt.pt - shift) / scale;
        normalizedKpts.push_back(normalizedKpt);
    }

    return normalizedKpts;
}

void RGBDHandler::compute_local_descriptors(
    std::shared_ptr<rtabmap::SensorData> &frame_data)
{
  // Extract local descriptors
  frame_data->uncompressData();
  std::vector<cv::KeyPoint> kpts_from;
  cv::Mat image = frame_data->imageRaw();

  cv::Mat depth_mask;
  if (!frame_data->depthRaw().empty())
  {
    if (image.rows % frame_data->depthRaw().rows == 0 &&
        image.cols % frame_data->depthRaw().cols == 0 &&
        image.rows / frame_data->depthRaw().rows ==
            frame_data->imageRaw().cols / frame_data->depthRaw().cols)
    {
      depth_mask = rtabmap::util2d::interpolate(
          frame_data->depthRaw(),
          frame_data->imageRaw().rows / frame_data->depthRaw().rows, 0.1f);
    }
    else
    {
      UWARN("%s is true, but RGB size (%dx%d) modulo depth size (%dx%d) is "
            "not 0. Ignoring depth mask for feature detection.",
            rtabmap::Parameters::kVisDepthAsMask().c_str(),
            frame_data->imageRaw().rows, frame_data->imageRaw().cols,
            frame_data->depthRaw().rows, frame_data->depthRaw().cols);
    }
  }

  auto extData = lightglueMatcher->Extractor(lightglueConfig, image);
  auto descriptors = extData.second;
  auto keypoints = std::move(extData.first);
  std::vector<cv::Point3f> kpts3D;
  if (!depth_mask.empty()) {
    //detector_->filterKeypointsByDepth(keypoints, depth_mask, 0.01f, 30.0f);
    kpts3D = detector_->generateKeypoints3D(*frame_data, keypoints);
  } else {
    kpts3D = detector_->generateKeypoints3D(*frame_data, keypoints);
    //detector_->filterKeypointsByDepth(keypoints, descriptors, kpts3D, 0.01f, 30.0f);
  }
  //RCLCPP_INFO(node_->get_logger(), "Data about things %d %d %d", keypoints.size(), descriptors.rows, kpts3D.size());
  frame_data->setFeatures(keypoints, kpts3D, descriptors);
}

bool RGBDHandler::setMatches(rtabmap::Signature &from, rtabmap::Signature &to) {
  
  auto origFrom = from.sensorData().keypoints(), origTo = to.sensorData().keypoints();
  auto kptsFrom3D = from.sensorData().keypoints3D(), kptsTo3D = to.sensorData().keypoints3D();
  cv::Mat descriptorsFrom; from.sensorData().descriptors().convertTo(descriptorsFrom, CV_32F);
  cv::Mat descriptorsTo; to.sensorData().descriptors().convertTo(descriptorsTo, CV_32F);
  std::list<int> fromWordIds;
  std::list<int> toWordIds;
  std::vector<int> fromWordIdsV(descriptorsFrom.rows);
  std::vector<int> toWordIdsV(descriptorsTo.rows, 0);
  for (int i = 0; i < descriptorsFrom.rows; ++i)
  {
    int id = i+1;
    fromWordIds.push_back(id);
    fromWordIdsV[i] = id;
  }

  auto fromModel = from.sensorData().stereoCameraModels().size() > 0? from.sensorData().stereoCameraModels()[0].left() : from.sensorData().cameraModels()[0];
  auto toModel = to.sensorData().stereoCameraModels().size() > 0? to.sensorData().stereoCameraModels()[0].left() : to.sensorData().cameraModels()[0];

  std::vector<cv::Point2f> kptsFrom = NormKeypoints(origFrom, fromModel.imageHeight(), fromModel.imageWidth());
  std::vector<cv::Point2f> kptsTo = NormKeypoints(origTo, toModel.imageHeight(), toModel.imageWidth());
  
  //RCLCPP_INFO(node_->get_logger(), "Data about things %d %d -- %d %d", kptsTo.size(), kptsFrom.size(), descriptorsTo.rows, descriptorsFrom.rows);

  auto matches = lightglueMatcher->MatcherIdx(lightglueConfig, kptsTo, kptsFrom, descriptorsTo, descriptorsFrom);
  //RCLCPP_INFO(node_->get_logger(), "Matches:  %d", matches.size());
  // if(matches.size() < 10) {
  //   return false;
  // }
    for(size_t i=0; i<matches.size(); ++i)
    {
        toWordIdsV[matches[i].queryIdx] = fromWordIdsV[matches[i].trainIdx];
    }
    for(size_t i=0; i<toWordIdsV.size(); ++i)
    {
        int toId = toWordIdsV[i];
        if(toId==0)
        {
            toId = fromWordIds.back()+i+1;
        }
        toWordIds.push_back(toId);
    }
    std::multiset<int> fromWordIdsSet(fromWordIds.begin(), fromWordIds.end());
    std::multiset<int> toWordIdsSet(toWordIds.begin(), toWordIds.end());

    std::multimap<int, int> wordsFrom;
    std::multimap<int, int> wordsTo;
    std::vector<cv::KeyPoint> wordsKptsFrom;
    std::vector<cv::KeyPoint> wordsKptsTo;
    std::vector<cv::Point3f> words3From;
    std::vector<cv::Point3f> words3To;

    int i=0;
    UASSERT(kptsFrom3D.empty() || fromWordIds.size() == kptsFrom3D.size());
    UASSERT(int(fromWordIds.size()) == descriptorsFrom.rows);
    for(std::list<int>::iterator iter=fromWordIds.begin(); iter!=fromWordIds.end(); ++iter)
    {
        if(fromWordIdsSet.count(*iter) == 1)
        {
          wordsFrom.insert(wordsFrom.end(), std::make_pair(*iter, wordsFrom.size()));
          if (!origFrom.empty())
          {
              wordsKptsFrom.push_back(origFrom[i]);
          }
          if(!kptsFrom3D.empty())
          {
              words3From.push_back(kptsFrom3D[i]);
          }
        }
        ++i;
    }
    UASSERT(kptsTo3D.size() == 0 || kptsTo3D.size() == kptsTo.size());
    UASSERT(toWordIds.size() == kptsTo.size());
    UASSERT(int(toWordIds.size()) == descriptorsTo.rows);

    i=0;
    for(std::list<int>::iterator iter=toWordIds.begin(); iter!=toWordIds.end(); ++iter)
    {
      if(toWordIdsSet.count(*iter) == 1)
      {
          wordsTo.insert(wordsTo.end(), std::make_pair(*iter, wordsTo.size()));
          wordsKptsTo.push_back(origTo[i]);
          if(!kptsTo3D.empty())
          {
              words3To.push_back(kptsTo3D[i]);
          }
      }
      ++i;
    }
    from.setWords(wordsFrom, wordsKptsFrom, words3From, cv::Mat());
    to.setWords(wordsTo, wordsKptsTo, words3To, cv::Mat());
    return true;
}

bool RGBDHandler::generate_new_keyframe(std::shared_ptr<rtabmap::SensorData> &keyframe)
{
  // Keyframe generation heuristic
  bool generate_new_keyframe = true;
  if (generate_new_keyframes_based_on_inliers_ratio_)
  {
    if (nb_local_keyframes_ > 0)
    {
      try
      {
        rtabmap::RegistrationInfo reg_info;
        auto from = Signature(*keyframe), to = Signature(*previous_keyframe_);
        setMatches(from, to);
        rtabmap::Transform t = intra_registration_.computeTransformation(
            from, to, rtabmap::Transform(), &reg_info);
        
        if (!t.isNull())
        {
          
          if (float(reg_info.inliers) >
              keyframe_generation_ratio_threshold_ *
                  float(previous_keyframe_->keypoints3D().size()))
          {
            generate_new_keyframe = false;
          }
        } 
        
        if (from.sensorData().keypoints3D().size() < 20) {
          //We have no depth data, this is a useless keyframe to us
          generate_new_keyframe = false;
        }
      }
      catch (std::exception &e)
      {
        RCLCPP_WARN(
            node_->get_logger(),
            "Exception: Could not compute transformation for keyframe generation: %s",
            e.what());
      }
    }
    if (generate_new_keyframe)
    {
      previous_keyframe_ = keyframe;
    }
  }
  // RCLCPP_WARN(
  //   node_->get_logger(),
  //   "Local transform: %s",
  //    keyframe->stereoCameraModels()[0].localTransform().prettyPrint().c_str());
 
  return generate_new_keyframe;
}

void RGBDHandler::process_new_sensor_data()
{
if (!received_data_queue_.empty())
  {
    auto sensor_data = received_data_queue_.front();
    received_data_queue_.pop_front();

    sensor_msgs::msg::NavSatFix gps_fix;
    if (enable_gps_recording_) {
      gps_fix = received_gps_queue_.front();
      received_gps_queue_.pop_front();
    }

    if(base_frame_id_.length() > 0) {
      if(!hasTransform_) {
          geometry_msgs::msg::TransformStamped t;

          // Look up for the transformation between target_frame and turtle2 frames
          // and send velocity commands for turtle2 to reach target_frame
          try {
            t = tf_buffer_->lookupTransform(
              sensor_data.second->child_frame_id, base_frame_id_, 
              sensor_data.second->header.stamp);
              
            tf2::fromMsg(t.transform, base_transform_);
            hasTransform_ = true;
          } catch (const tf2::TransformException & ex) {
            RCLCPP_INFO(
              node_->get_logger(), "Could not transform %s to %s: %s",
              sensor_data.second->child_frame_id.c_str(), base_frame_id_.c_str(), ex.what());
          }
      }
    }


    if (sensor_data.first->isValid())
    {
      // Compute local descriptors
      compute_local_descriptors(sensor_data.first);

      bool generate_keyframe = generate_new_keyframe(sensor_data.first);
      if (generate_keyframe)
      {
        // Set keyframe ID
        sensor_data.first->setId(nb_local_keyframes_);
        nb_local_keyframes_++;

        if (enable_gps_recording_) {
          send_keyframe(sensor_data, &gps_fix);
        } else {
          // Send keyframe for loop detection
          send_keyframe(sensor_data);
        }
      }

      clear_sensor_data(sensor_data.first);

      if (generate_keyframe)
      {
        //Reduce our descriptor size here for easier storage and transmission
        cv::Mat fp16descriptors;
        sensor_data.first->descriptors().convertTo(fp16descriptors, CV_16F);
        //Maybe do this as well .... depends on what performance looks like rtabmap::compressData2(fp16descriptors);
        sensor_data.first->setFeatures(sensor_data.first->keypoints(), sensor_data.first->keypoints3D(), fp16descriptors);
        local_descriptors_map_.insert({sensor_data.first->id(), sensor_data.first});
      }
    }
  }

}

void RGBDHandler::sensor_data_to_rgbd_msg(
    const std::shared_ptr<rtabmap::SensorData> sensor_data,
    rtabmap_msgs::msg::SensorData &msg_data, bool baselinkFrame)
{
  rtabmap_conversions::sensorDataToROS(*sensor_data, msg_data);
  if (baselinkFrame) {
    rtabmap_conversions::points3fToROS(sensor_data->keypoints3D(), msg_data.points);
  }
}

void RGBDHandler::local_descriptors_request(
    cslam_common_interfaces::msg::LocalDescriptorsRequest::
        ConstSharedPtr request)
{
  // Fill msg
  auto msg = std::make_unique<cslam_common_interfaces::msg::LocalImageDescriptors>();

  sensor_data_to_rgbd_msg(local_descriptors_map_.at(request->keyframe_id),
                          msg->data);
  msg->keyframe_id = request->keyframe_id;
  msg->robot_id = robot_id_;
  msg->matches_robot_id = request->matches_robot_id;
  msg->matches_keyframe_id = request->matches_keyframe_id;

  // Publish local descriptors
  local_descriptors_publisher_->publish(std::move(msg));

  if (enable_logs_)
  {
    log_total_local_descriptors_cumulative_communication_ += msg->data.key_points.size()*28; // bytes
    log_total_local_descriptors_cumulative_communication_ += msg->data.points.size()*12; // bytes
    log_total_local_descriptors_cumulative_communication_ += msg->data.descriptors.size(); // bytes
    diagnostic_msgs::msg::KeyValue log_msg;
    log_msg.key = "local_descriptors_cumulative_communication";
    log_msg.value = std::to_string(log_total_local_descriptors_cumulative_communication_);
    log_publisher_->publish(log_msg);
  }
}

void RGBDHandler::receive_local_keyframe_match(
    cslam_common_interfaces::msg::LocalKeyframeMatch::ConstSharedPtr
        msg)
{
  try
  {
    auto keyframe0 = local_descriptors_map_.at(msg->keyframe0_id);
    auto keyframe1 = local_descriptors_map_.at(msg->keyframe1_id);
    rtabmap::RegistrationInfo reg_info;
    auto from = Signature(*keyframe0), to = Signature(*keyframe1);
    auto lc = std::make_unique<cslam_common_interfaces::msg::IntraRobotLoopClosure>();
    lc->keyframe0_id = msg->keyframe0_id;
    lc->keyframe1_id = msg->keyframe1_id;
    
    setMatches(from, to);
    rtabmap::Transform t = intra_registration_.computeTransformation(
      from, to, rtabmap::Transform(), &reg_info);
    
    if (!t.isNull())
    {
      lc->success = true;
      auto fluFrame = CameraModel::opticalRotation() * t * CameraModel::opticalRotation().inverse();
      //RCLCPP_INFO(node_->get_logger(), "Intra loop closure: %s", t.prettyPrint().c_str());
      reg_info.covariance.reshape(1,1).copyTo(lc->pose.covariance);
      rtabmap_conversions::transformToPoseMsg(fluFrame, lc->pose.pose);
    }
    else
    {
      lc->success = false;
    }
    intra_robot_loop_closure_publisher_->publish(std::move(lc));
  }
  catch (std::exception &e)
  {
    RCLCPP_WARN(
        node_->get_logger(),
        "Exception: Could not compute local transformation between %d and %d: %s",
        msg->keyframe0_id, msg->keyframe1_id,
        e.what());
  }
}

void RGBDHandler::local_descriptors_msg_to_sensor_data(
    const std::shared_ptr<
        cslam_common_interfaces::msg::LocalImageDescriptors>
        msg,
    rtabmap::SensorData &sensor_data)
{
  // // Fill descriptors
  // rtabmap::CameraModel camera_model =
  //     rtabmap_conversions::cameraModelFromROS(msg->data.rgb_camera_info,
  //                                     rtabmap::Transform::getIdentity());
  // sensor_data = rtabmap::SensorData(
  //     cv::Mat(), cv::Mat(), camera_model, 0,
  //     rtabmap_conversions::timestampFromROS(msg->data.header.stamp));

  // std::vector<cv::KeyPoint> kpts;
  // rtabmap_conversions::keypointsFromROS(msg->data.key_points, kpts);
  // std::vector<cv::Point3f> kpts3D;
  // rtabmap_conversions::points3fFromROS(msg->data.points, kpts3D);
  // auto descriptors = rtabmap::uncompressData(msg->data.descriptors);
  // sensor_data.setFeatures(kpts, kpts3D, descriptors);
  sensor_data = rtabmap_conversions::sensorDataFromROS(msg->data);
}

void RGBDHandler::receive_local_image_descriptors(
    const std::shared_ptr<
        cslam_common_interfaces::msg::LocalImageDescriptors>
        msg)
{
  std::deque<int> keyframe_ids;
  for (unsigned int i = 0; i < msg->matches_robot_id.size(); i++)
  {
    if (msg->matches_robot_id[i] == robot_id_)
    {
      keyframe_ids.push_back(msg->matches_keyframe_id[i]);
    }
  }

  for (auto local_keyframe_id : keyframe_ids)
  {
    try
    {
      rtabmap::SensorData tmp_to;
      local_descriptors_msg_to_sensor_data(msg, tmp_to);

      // Compute transformation
      //  Registration params
      rtabmap::RegistrationInfo reg_info;
      auto tmp_from = local_descriptors_map_.at(local_keyframe_id);
      auto from = Signature(*tmp_from), to = Signature(tmp_to);
      setMatches(from, to);
      rtabmap::Transform t = inter_registration_.computeTransformation(
        from, to, rtabmap::Transform(), &reg_info);
      
      // Store using pairs (robot_id, keyframe_id)
      auto lc = std::make_unique<cslam_common_interfaces::msg::InterRobotLoopClosure>();
      lc->robot0_id = robot_id_;
      lc->robot0_keyframe_id = local_keyframe_id;
      lc->robot1_id = msg->robot_id;
      lc->robot1_keyframe_id = msg->keyframe_id;
      if (!t.isNull())
      {
        lc->success = true;
        auto fluFrame = CameraModel::opticalRotation() * t * CameraModel::opticalRotation().inverse();
        reg_info.covariance.reshape(1,1).copyTo(lc->pose.covariance);
        rtabmap_conversions::transformToPoseMsg(fluFrame, lc->pose.pose);
        inter_robot_loop_closure_publisher_->publish(std::move(lc));
      }
      else
      {
        RCLCPP_INFO(
            node_->get_logger(),
            "Could not compute transformation between (%d,%d) and (%d,%d): %s",
            robot_id_, local_keyframe_id, msg->robot_id, msg->keyframe_id,
            reg_info.rejectedMsg.c_str());
        lc->success = false;
        inter_robot_loop_closure_publisher_->publish(std::move(lc));
      }
    }
    catch (std::exception &e)
    {
      RCLCPP_WARN(
          node_->get_logger(),
          "Exception: Could not compute transformation between (%d,%d) and (%d,%d): %s",
          robot_id_, local_keyframe_id, msg->robot_id, msg->keyframe_id,
          e.what());
    }
  }
}

const Eigen::Affine3d opticalTransform = rtabmap::CameraModel::opticalRotation().toEigen3d();

void RGBDHandler::send_keyframe(const std::pair<std::shared_ptr<rtabmap::SensorData>, std::shared_ptr<const nav_msgs::msg::Odometry>> &keypoints_data, const sensor_msgs::msg::NavSatFix * gps_data)
{
  cv::Mat img;
  if (global_image_topic_.length() > 0) {
    keypoints_data.first->uncompressDataConst(0, 0, 0, &img);
  } else {
    keypoints_data.first->uncompressDataConst(&img, 0);
  }

  // Image message
  std_msgs::msg::Header header;
  header.stamp = keypoints_data.second->header.stamp;
  cv_bridge::CvImage image_bridge = cv_bridge::CvImage(header, img.channels() > 1? "rgb8":"mono8", img);
  auto keyframe_msg = std::make_unique<cslam_common_interfaces::msg::KeyframeRGB>();
  image_bridge.toImageMsg(keyframe_msg->image);
  keyframe_msg->id = keypoints_data.first->id();

  keyframe_data_publisher_->publish(std::move(keyframe_msg));

  auto odom_msg = std::make_unique<cslam_common_interfaces::msg::KeyframeOdom>() ;
  odom_msg->id = keypoints_data.first->id();
  odom_msg->odom = *keypoints_data.second;
  if(gps_data != nullptr) 
    odom_msg->gps = *gps_data;

  if(base_frame_id_.length() > 0 && hasTransform_) {

      tf2::Transform poseTf;
      tf2::fromMsg(odom_msg->odom.pose.pose, poseTf);
      auto out = poseTf * base_transform_;
      tf2::toMsg(out, odom_msg->odom.pose.pose);
  }

  keyframe_odom_publisher_->publish(std::move(odom_msg));

  if (enable_visualization_)
  {
    send_visualization(keypoints_data);
  }
}

void RGBDHandler::send_visualization(const std::pair<std::shared_ptr<rtabmap::SensorData>, std::shared_ptr<const nav_msgs::msg::Odometry>> &keypoints_data)
{
  send_visualization_keypoints(keypoints_data);
  send_visualization_pointcloud(keypoints_data.first);
}

void RGBDHandler::clear_sensor_data(std::shared_ptr<rtabmap::SensorData>& sensor_data)
{
  // Clear costly data
  sensor_data->clearCompressedData();
  sensor_data->clearRawData();
}

void RGBDHandler::send_visualization_keypoints(const std::pair<std::shared_ptr<rtabmap::SensorData>, std::shared_ptr<const nav_msgs::msg::Odometry>> &keypoints_data)
{
  // visualization message
  auto features_msg = std::make_unique<cslam_common_interfaces::msg::LocalImageDescriptors>();
  features_msg->keyframe_id = keypoints_data.first->id();
  features_msg->robot_id = robot_id_;
  rtabmap_conversions::points3fToROS(keypoints_data.first->keypoints3D(), features_msg->data.points);
  features_msg->data.descriptors.clear();

  // Publish local descriptors
  visualization_local_descriptors_publisher_->publish(std::move(features_msg));
}

sensor_msgs::msg::PointCloud2 RGBDHandler::visualization_pointcloud_voxel_subsampling(
    const sensor_msgs::msg::PointCloud2 &input_cloud)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(input_cloud, *cloud);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(visualization_voxel_size_, visualization_voxel_size_, visualization_voxel_size_);
  sor.filter(*cloud_filtered);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_clipped(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud (cloud_filtered);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, visualization_max_range_);
  pass.filter(*cloud_filtered_clipped);

  sensor_msgs::msg::PointCloud2 output_cloud;
  pcl::toROSMsg(*cloud_filtered_clipped, output_cloud);
  output_cloud.header = input_cloud.header;
  return output_cloud;
}

void RGBDHandler::send_visualization_pointcloud(const std::shared_ptr<rtabmap::SensorData> & sensor_data)
{
  cslam_common_interfaces::msg::VizPointCloud keyframe_pointcloud_msg;
  keyframe_pointcloud_msg.robot_id = robot_id_;
  keyframe_pointcloud_msg.keyframe_id = sensor_data->id();
  std_msgs::msg::Header header;
  header.stamp = node_->now();
  header.frame_id = MAP_FRAME_ID(robot_id_);
  auto pointcloud_msg = create_colored_pointcloud(sensor_data, header);

  if (visualization_voxel_size_ > 0.0)
  {
    pointcloud_msg = visualization_pointcloud_voxel_subsampling(pointcloud_msg);
  }

  keyframe_pointcloud_msg.pointcloud = pointcloud_msg;
  keyframe_pointcloud_publisher_->publish(keyframe_pointcloud_msg);
}

void RGBDHandler::gps_callback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr msg)
{
  latest_gps_fix_ = *msg;
}
