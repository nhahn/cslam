#include <string>
#include "cslam/front_end/map_manager.h"
#include <rtabmap/utilite/ULogger.h>
#include "cslam/profiler.h"
#include <opencv2/features2d.hpp>
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
#include <future>
#include "rtabmap_msgs/msg/sensor_data.hpp"

#define MAP_FRAME_ID(id) "robot" + std::to_string(id) + "_map"

using namespace cslam;
using namespace rtabmap;

std::map<std::string, ULogger::Level> rtabmapLogLevel =
{
    { "debug", ULogger::kDebug },
    { "info", ULogger::kInfo },
    { "warning", ULogger::kWarning },
    { "error", ULogger::kError },
    { "fatal", ULogger::kFatal }
};

MapManager::MapManager(rclcpp::NodeOptions ops) : Node("map_manager", ops.start_parameter_event_publisher(false).start_parameter_services(false)), workerPool(3)
  {
    declare_parameter<std::string>("rtabmap.log_level", "warning");
    ULogger::setType(ULogger::kTypeConsole);
    auto level = rtabmapLogLevel.find(get_parameter("rtabmap.log_level").as_string());
    ULogger::setLevel(level == rtabmapLogLevel.end()? ULogger::kWarning : level->second);
    declare_parameter<int>("frontend.pnp_min_inliers", 20);
    declare_parameter<int>("frontend.min_3d_keypoints", 100);
    declare_parameter<int>("frontend.inter_pnp_min_inliers", get_parameter("frontend.pnp_min_inliers").as_int());
    declare_parameter<int>("frontend.intra_pnp_min_inliers", get_parameter("frontend.pnp_min_inliers").as_int());
    declare_parameter<int>("frontend.max_queue_size", 10);
    declare_parameter<int>("frontend.optFlow.maxKeypoints", 1024);
    declare_parameter<int>("frontend.optFlow.pyrLevels", 3);
    declare_parameter<int>("frontend.optFlow.iterations", 15);
    declare_parameter<int>("frontend.optFlow.windowSize", 11);
    declare_parameter<bool>("frontend.optFlow.usePVA", false);
    declare_parameter<std::string>("tf_prefix", "");
    declare_parameter<std::string>("frontend.base_frame", "base_link");
    declare_parameter<std::string>("frontend.odom_frame", "odom");
    declare_parameter<int>("max_nb_robots", 1);
    declare_parameter<int>("robot_id", 0);
    declare_parameter<int>("frontend.map_manager_process_period_ms", 100);
    declare_parameter<std::string>("frontend.sensor_type", "stereo");
    declare_parameter<std::string>("frontend.sync_method", "exact");
    declare_parameter<bool>("visualization.enable", false);
    declare_parameter<int>("visualization.publishing_period_ms", 0);
    declare_parameter<float>("visualization.voxel_size", 0.05);
    declare_parameter<float>("visualization.max_range", 2.0);
    declare_parameter<bool>("evaluation.enable_gps_recording", false);
    declare_parameter<std::string>("evaluation.gps_topic", "");
    declare_parameter<bool>("frontend.use_external_odom", true);

    declare_parameter<std::string>("frontend.superpoint_model", "/models/superpoint_1024.onnx");
    declare_parameter<std::string>("frontend.lightglue_model", "/models/superpoint_lightglue_1024.onnx");
    declare_parameter<float>("frontend.keyframe_generation_ratio_threshold", 0.0);
    declare_parameter<std::string>("frontend.sensor_base_frame_id", ""); // If empty we assume that the camera link is the base link
    declare_parameter<bool>("evaluation.enable_logs", false);
    declare_parameter<std::string>("frontend.global_descriptor_image_topic", "");

    get_parameter("frontend.sensor_type", sensor_type);
    auto sync_type = get_parameter("frontend.sync_method").as_string();
    if (sensor_type == "stereo") {
      if (sync_type == "exact")
        sensor_handler_ = std::make_shared<StereoHandler<ExactStereoSync>>(this);
      else
        sensor_handler_ = std::make_shared<StereoHandler<ApproximateStereoSync>>(this);
    } 
    else if (sensor_type == "rgbd") {
      if (sync_type == "exact")
        sensor_handler_ = std::make_shared<RGBDHandler<ExactRGBDSync>>(this);
      else
        sensor_handler_ = std::make_shared<RGBDHandler<ApproximateRGBDSync>>(this);
    } 
    else {
        RCLCPP_ERROR(get_logger(), "Sensor type not supported: %s",
                    sensor_type.c_str());
    }

    get_parameter("frontend.map_manager_process_period_ms",
                        map_manager_process_period_ms_);
      // Parameters
    get_parameter("frontend.inter_pnp_min_inliers", min_inliers_);
    get_parameter("max_nb_robots", max_nb_robots_);
    get_parameter("robot_id", robot_id_);
    get_parameter("frontend.max_queue_size", max_queue_size_);
    get_parameter("frontend.min_3d_keypoints", min_3d_keypoints_);
    get_parameter("frontend.keyframe_generation_ratio_threshold", keyframe_generation_ratio_threshold_);
    get_parameter("visualization.enable",
                        enable_visualization_);
    get_parameter("visualization.publishing_period_ms",
                        visualization_period_ms_);
    declare_parameter("frontend.matcher_threshold", 0.1f);
    get_parameter("visualization.voxel_size",
                        visualization_voxel_size_);
    get_parameter("visualization.max_range",
                        visualization_max_range_);

    get_parameter("evaluation.enable_logs",
                        enable_logs_);


    global_image_topic_ = get_parameter("frontend.global_descriptor_image_topic").as_string();

    //Fetch all the rtabmap parameters and then assign them to an rtabmap param setup
    //Initialize the interface, or we get an error
    std::map<std::string, rclcpp::Parameter> pmap;
    get_parameters("rtabmap", pmap);
    auto paramList = rtabmap::Parameters::getDefaultParameters();
    for (auto const& x : paramList) {
      declare_parameter<std::string>("rtabmap." + x.first, x.second);
      auto val = get_parameter("rtabmap." + x.first);
      rtabmap_parameters.insert_or_assign(x.first, val.as_string());
    }
    
    detector_ = rtabmap::Feature2D::create(rtabmap_parameters);
    detector_->parseParameters(rtabmap_parameters);
    lightglueConfig = lightglue::Configuration{
      get_parameter("frontend.superpoint_model").as_string(),
      get_parameter("frontend.lightglue_model").as_string()
    };
    lightglueConfig.grayScale = true;
    lightglueConfig.matcherUseTrt = true;
    lightglueConfig.extractorUseTrt = true;

    std::chrono::milliseconds period(map_manager_process_period_ms_);
    sensorDataCB = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    process_timer_ = create_wall_timer(
        std::chrono::milliseconds(period),
        std::bind(&MapManager::process_new_sensor_data, this), sensorDataCB);

    nb_local_keyframes_ = 0;
    odom_status = get_parameter("frontend.use_external_odom").as_bool()? OdomState::EXTERNAL : OdomState::GLOBAL_TRACKING;
    if (odom_status != EXTERNAL) {
     rclcpp::SubscriptionOptions recoveryOptions;
     recoveryOptions.callback_group = sensorDataCB;
    
     odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>(get_parameter("frontend.odom_topic").as_string() + "/raw", 5);
     recovery_subscriber_ = create_subscription<cslam_common_interfaces::msg::InterRobotMatches>("cslam/odom_recovery", 1, std::bind(&MapManager::recover_odom_pose, this, std::placeholders::_1), recoveryOptions);
     add_recovered_publisher_ = create_publisher<cslam_common_interfaces::msg::LocalKeyframeMatch>("cslam/add_recovered_pose", 5);
     tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
     
    }

    calcOdom.header.frame_id = get_parameter("tf_prefix").as_string() + get_parameter("frontend.odom_frame").as_string();
    calcOdom.child_frame_id = get_parameter("tf_prefix").as_string() + get_parameter("frontend.base_frame").as_string();
    odomTf.header.frame_id = calcOdom.header.frame_id;
    odomTf.child_frame_id = calcOdom.child_frame_id;
    // Service to extract and publish local image descriptors to another robot
    rclcpp::SubscriptionOptions descriptorOptions;
    descriptorOptions.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    send_local_descriptors_subscriber_ = create_subscription<
        cslam_common_interfaces::msg::LocalDescriptorsRequest>(
        "cslam/local_descriptors_request", 100,
        std::bind(&MapManager::local_descriptors_request, this,
                  std::placeholders::_1), descriptorOptions);


    // Publisher for global descriptors
    keyframe_data_publisher_ =
        create_publisher<cslam_common_interfaces::msg::KeyframeRGB>(
            "cslam/keyframe_data", 100);

    // Publisher for odometry with ID
    keyframe_odom_publisher_ =
        create_publisher<cslam_common_interfaces::msg::KeyframeOdom>(
            "cslam/keyframe_odom", 100);

    keyframe_keypoint_viz_ =
        create_publisher<sensor_msgs::msg::Image>(
            "cslam/viz/keyframe_keypoints", 100);

    // keyframe_matches_viz_ =
    //     create_publisher<sensor_msgs::msg::Image>(
    //         "cslam/viz/keyframe_matches", 100);

    rclcpp::SubscriptionOptions intraOptions;
    intraOptions.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    // Local matches subscription
    local_keyframe_match_subscriber_ = create_subscription<
        cslam_common_interfaces::msg::LocalKeyframeMatch>(
        "cslam/local_keyframe_match", 100,
        std::bind(&MapManager::receive_local_keyframe_match, this,
                  std::placeholders::_1), intraOptions);

    // Publishers to other robots local descriptors subscribers
    std::string local_descriptors_topic = "/cslam/local_descriptors";
    local_descriptors_publisher_ = create_publisher<
        cslam_common_interfaces::msg::LocalImageDescriptors>(local_descriptors_topic, 100);

    if (enable_visualization_)
    {
      visualization_local_descriptors_publisher_ = create_publisher<
          cslam_common_interfaces::msg::LocalImageDescriptors>("/cslam/viz/local_descriptors", 100);

      keyframe_pointcloud_publisher_ = create_publisher<cslam_common_interfaces::msg::VizPointCloud>(
          "/cslam/viz/keyframe_pointcloud", 100);
    }

    rclcpp::SubscriptionOptions interOptions;
    interOptions.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // Subscriber for local descriptors
    local_descriptors_subscriber_ = create_subscription<
        cslam_common_interfaces::msg::LocalImageDescriptors>(
        "/cslam/local_descriptors", 100,
        std::bind(&MapManager::receive_local_image_descriptors, this,
                  std::placeholders::_1), interOptions);

    // Registration settings
    auto interParams = rtabmap::ParametersMap(rtabmap_parameters);
    interParams.insert_or_assign(rtabmap::Parameters::kVisMinInliers(), std::to_string(get_parameter("frontend.inter_pnp_min_inliers").as_int()));
    interParams.insert_or_assign(rtabmap::Parameters::kVisForwardEstOnly(), "true");
    interParams.insert_or_assign(rtabmap::Parameters::kVisRefineIterations(), "0");
    interParams.insert_or_assign(rtabmap::Parameters::kVisIterations(), "900");
    interParams.insert_or_assign(rtabmap::Parameters::kVisPnPVarianceMedianRatio(), "3");
    inter_registration_ = std::make_shared<rtabmap::RegistrationVis>(interParams);

    auto intraParams = rtabmap::ParametersMap(rtabmap_parameters);
    intraParams.insert_or_assign(rtabmap::Parameters::kVisMinInliers(), std::to_string(get_parameter("frontend.intra_pnp_min_inliers").as_int()));
    intraParams.insert_or_assign(rtabmap::Parameters::kVisForwardEstOnly(), "false");
    intra_registration_ = std::make_shared<rtabmap::RegistrationVis>(intraParams);

    auto f2fParams = rtabmap::ParametersMap(rtabmap_parameters);
    f2fParams.insert_or_assign(rtabmap::Parameters::kVisIterations(), "300");
    f2fParams.insert_or_assign(rtabmap::Parameters::kVisMinInliers(), "60");
    f2fParams.insert_or_assign(rtabmap::Parameters::kVisPnPRefineIterations(), "0");
    f2f_registration_ = std::make_shared<rtabmap::RegistrationVis>(f2fParams);

    auto ofKeypoints = get_parameter("frontend.optFlow.maxKeypoints").as_int();
    auto pyrLevels = get_parameter("frontend.optFlow.pyrLevels").as_int();
    auto iterations = get_parameter("frontend.optFlow.iterations").as_int();
    auto windowSize = get_parameter("frontend.optFlow.windowSize").as_int();
    auto usePVA = get_parameter("frontend.optFlow.usePVA").as_bool();

    optical_matcher = std::make_shared<OpticalFlow>(ofKeypoints, pyrLevels, iterations, windowSize, usePVA);
    // Intra-robot loop closure publisher
    intra_robot_loop_closure_publisher_ = create_publisher<
        cslam_common_interfaces::msg::IntraRobotLoopClosure>(
        "cslam/intra_robot_loop_closure", 100);

    // Publisher for inter robot loop closure to all robots
    inter_robot_loop_closure_publisher_ = create_publisher<
        cslam_common_interfaces::msg::InterRobotLoopClosure>(
        "/cslam/inter_robot_loop_closure", 100);

    if (enable_logs_){
      log_total_local_descriptors_cumulative_communication_ = 0;
      log_publisher_ = create_publisher<diagnostic_msgs::msg::KeyValue>(
          "cslam/log_info", 100);
    }

    lastKFPose.setIdentity();

  RCLCPP_INFO(get_logger(), "Initialization done.");
}



bool MapManager::compute_local_descriptors(
    std::shared_ptr<rtabmap::SensorData> frame_data,
    const cv::Mat &img)
{
  PROFILE_ME;

  try {
    
    auto extData = lightglueMatcher->Extractor(lightglueConfig, img, frame_data->depthRaw());
    std::vector<cv::Point3f> kpts3D = detector_->generateKeypoints3D(*frame_data, extData.first);
    int valid3DKpts = 0;
    for(size_t i = 0; i < kpts3D.size(); i++) {
      if(rtabmap::util3d::isFinite(kpts3D[i])) {
        valid3DKpts++;
      }
    }

    if (keyframe_keypoint_viz_->get_subscription_count() > 0) {
      auto imgMsg = std::make_unique<sensor_msgs::msg::Image>();
      cv::drawKeypoints(img, extData.first, keypointViz);
      std_msgs::msg::Header header;
      header.frame_id = sensor_handler_->sensor_frame;
      header.stamp = now();
      cv_bridge::CvImage(header, "bgr8", keypointViz).toImageMsg(*imgMsg);
      keyframe_keypoint_viz_->publish(std::move(imgMsg));
    }

    if(valid3DKpts < min_3d_keypoints_){
      RCLCPP_DEBUG(get_logger(), "Rejecting keyframe due to the low number of 3D keypoints detected (%d/%lu) - min %d", valid3DKpts, extData.first.size(), min_3d_keypoints_);
      return false;
    }
    //Reduce our descriptor size here for easier storage and transmission
    cv::Mat fp16descriptors;
    extData.second.convertTo(fp16descriptors, CV_16F);
    //RCLCPP_INFO(get_logger(), "Data about things %d %d %d", keypoints.size(), descriptors.rows, kpts3D.size());
    frame_data->setFeatures(extData.first, kpts3D, fp16descriptors);
  } catch (std::exception &e) {
    RCLCPP_ERROR(get_logger(),"Error extracting keypoints for keyframe %s", e.what());
  }

  return true;
}

bool MapManager::setMatches(rtabmap::Signature &from, rtabmap::Signature &to) {
  PROFILE_ME;

  const auto kptsFrom = from.sensorData().keypoints(), kptsTo = to.sensorData().keypoints();
  const auto kptsFrom3D = from.sensorData().keypoints3D(), kptsTo3D = to.sensorData().keypoints3D();
  cv::Mat descriptorsFrom; from.sensorData().descriptors().convertTo(descriptorsFrom, CV_32F);
  cv::Mat descriptorsTo; to.sensorData().descriptors().convertTo(descriptorsTo, CV_32F);
  const auto fromModel = from.sensorData().stereoCameraModels().size() > 0? from.sensorData().stereoCameraModels()[0].left() : from.sensorData().cameraModels()[0];
  const auto toModel = to.sensorData().stereoCameraModels().size() > 0? to.sensorData().stereoCameraModels()[0].left() : to.sensorData().cameraModels()[0];

  std::vector<cv::DMatch> matches;
  try {
    //Query = TO keypoints, Train = FROM Keypoints
    matches = lightglueMatcher->Matcher(lightglueConfig, kptsTo, kptsFrom, descriptorsTo, descriptorsFrom, toModel.imageSize(), fromModel.imageSize());
  } catch (std::exception &e) {
    RCLCPP_ERROR(get_logger(),"Error matching KFs (%d,%d) - %s", from.id(), to.id(), e.what());
  }
  
  if(matches.size() == 0) {
    return false;
  }

  std::multimap<int, int> wordsFrom;
  std::multimap<int, int> wordsTo;
  std::vector<cv::KeyPoint> wordsKptsFrom;
  std::vector<cv::KeyPoint> wordsKptsTo;
  std::vector<cv::Point3f> words3From;
  std::vector<cv::Point3f> words3To;
  std::vector<cv::KeyPoint> kptsFromKept(matches.size());
  std::vector<cv::KeyPoint> kptsToKept(matches.size());
  std::vector<cv::Point3f> kptsTo3DKept(matches.size());
  std::vector<cv::Point3f> kptsFrom3DKept(matches.size());
  for(unsigned int i=0; i<matches.size(); ++i)
  {
      kptsFromKept[i] = kptsFrom[matches[i].trainIdx];
      kptsFrom3DKept[i] = kptsFrom3D[matches[i].trainIdx];
      kptsToKept[i] = kptsTo[matches[i].queryIdx];
      kptsTo3DKept[i] = kptsTo3D[matches[i].queryIdx];
  }

  UASSERT(kptsFromKept.size() == kptsFrom3DKept.size());
  UASSERT(kptsFromKept.size() == kptsToKept.size());
  for(unsigned int i=0; i< kptsFrom3DKept.size(); ++i)
  {
    wordsFrom.insert(wordsFrom.end(), std::make_pair(i, wordsFrom.size()));
    wordsKptsFrom.push_back(kptsFromKept[i]);
    words3From.push_back(kptsFrom3DKept[i]);

    wordsTo.insert(wordsTo.end(), std::make_pair(i, wordsTo.size()));
    wordsKptsTo.push_back(kptsToKept[i]);
    words3To.push_back(kptsTo3DKept[i]);
  }

  RCLCPP_DEBUG(get_logger(), "Found the following matches: Words %lu -> %lu, Kpts %lu -> %lu, 3D Kpts %lu -> %lu", wordsFrom.size(), wordsTo.size(), wordsKptsFrom.size(),
      wordsKptsTo.size(), words3From.size(), words3To.size());
  from.setWords(wordsFrom, wordsKptsFrom, words3From, cv::Mat());
  to.setWords(wordsTo, wordsKptsTo, words3To, cv::Mat());
  return true;
}

std::pair<std::shared_ptr<rtabmap::Signature>, std::shared_ptr<rtabmap::Signature>> MapManager::computeMatches(const rtabmap::SensorData& f, const rtabmap::SensorData& t) {
    PROFILE_ME;

    auto from = std::make_shared<rtabmap::Signature>(f), to = std::make_shared<rtabmap::Signature>(t);
    from->sensorData().clearRawData(); to->sensorData().clearRawData();
    from->sensorData().clearCompressedData(); to->sensorData().clearCompressedData();
    bool hasMaches = setMatches(*from, *to);
    if (!hasMaches)
      return std::pair<std::shared_ptr<rtabmap::Signature>, std::shared_ptr<rtabmap::Signature>>(nullptr, nullptr);
    else return std::make_pair(from, to);
}

rtabmap::Transform MapManager::compute_flow(const std::shared_ptr<rtabmap::SensorData> newData, const cv::Mat& toImg, rtabmap::RegistrationInfo &reg_info)
{
    PROFILE_ME;
    auto from = std::make_shared<rtabmap::Signature>(*current_OF_frame_), to = std::make_shared<rtabmap::Signature>(*newData);
    from->sensorData().clearRawData();     to->sensorData().clearRawData();
    from->sensorData().clearCompressedData(); to->sensorData().clearCompressedData();
    std::multimap<int, int> wordsFrom;
    std::multimap<int, int> wordsTo;
    std::vector<cv::KeyPoint> wordsKptsFrom;
    std::vector<cv::KeyPoint> wordsKptsTo;
    std::vector<cv::Point3f> words3From;
    std::vector<cv::Point3f> words3To;
    std::vector<cv::KeyPoint> kptsTo(current_OF_frame_->keypoints().size());
    std::vector<cv::KeyPoint> kptsFrom = current_OF_frame_->keypoints();
    std::vector<cv::Point3f> kptsFrom3D = current_OF_frame_->keypoints3D();
    std::vector<cv::Point3f> kptsFrom3DKept(current_OF_frame_->keypoints3D().size());
    int ki = 0;
    auto matches = optical_matcher->matchNextFrame(toImg);
    if (!matches.size()) return rtabmap::Transform();

    for(unsigned int i=0; i<matches.size(); ++i)
    {
      if(matches[i].first &&
          uIsInBounds(matches[i].second.x, 0.0f, float(toImg.cols)) &&
          uIsInBounds(matches[i].second.y, 0.0f, float(toImg.rows)))
      {
        kptsFrom[ki] = cv::KeyPoint(current_OF_frame_->keypoints()[i].pt, 1);
        kptsFrom3DKept[ki] = kptsFrom3D[i];
        kptsTo[ki++] = cv::KeyPoint(matches[i].second, 1);
      }
    }
    RCLCPP_DEBUG(
      get_logger(),
      "Optical flow matches: %d", ki);

    if (ki < f2f_registration_->getMinInliers()) {
      return rtabmap::Transform();
    }

    kptsFrom.resize(ki);
    kptsTo.resize(ki);
    kptsFrom3DKept.resize(ki);
    kptsFrom3D = kptsFrom3DKept;

    UASSERT(kptsFrom.size() == kptsFrom3DKept.size());
    UASSERT(kptsFrom.size() == kptsTo.size());
    for(unsigned int i=0; i< kptsFrom3DKept.size(); ++i)
    {
      wordsFrom.insert(wordsFrom.end(), std::make_pair(i, wordsFrom.size()));
      wordsKptsFrom.push_back(kptsFrom[i]);
      words3From.push_back(kptsFrom3DKept[i]);

      wordsTo.insert(wordsTo.end(), std::make_pair(i, wordsTo.size()));
      wordsKptsTo.push_back(kptsTo[i]);
    }
    from->setWords(wordsFrom, wordsKptsFrom, words3From, cv::Mat());
    to->setWords(wordsTo, wordsKptsTo, words3To, cv::Mat());
    try
    {
        return f2f_registration_->computeTransformation(
          *from, *to, rtabmap::Transform(), &reg_info);
    }
    catch (std::exception &e)
    {
      RCLCPP_WARN(
          get_logger(),
          "Exception: OF transform not computable: %s -- from words %lu words3 %lu : to words %lu words3 %lu",
          e.what(), from->getWords().size(), from->getWords3().size(), to->getWords().size(), to->getWords3().size());
    }
    return rtabmap::Transform();
  }

void MapManager::publish_odom_update(const rtabmap::Transform &pose, const cv::Mat &covariance) {
  if (odom_status == GLOBAL_TRACKING) {
    covariance.reshape(1,1).copyTo(calcOdom.pose.covariance);
    rtabmap_conversions::transformToPoseMsg(pose, calcOdom.pose.pose);
    odom_publisher_->publish(calcOdom);
  }
}

void MapManager::process_new_sensor_data()
{
  PROFILE_ME;

  auto pair = sensor_handler_->getNextPair();
  auto sensor_data = pair.first;
  auto odom = pair.second;
  if (!sensor_data) return;
  if (!lightglueMatcher) { //On our first bit of sensor data, initialize the lightglue matcher with the image dimensions
    lightglueMatcher = std::make_shared<lightglue::LightGlueOnnxRunner>();
    lightglueConfig.extractorImageDims.width = sensor_data->imageRaw().cols;
    lightglueConfig.extractorImageDims.height = sensor_data->imageRaw().rows;
    lightglueMatcher->InitOrtEnv(lightglueConfig);
    lightglueMatcher->SetMatchThresh(get_parameter("frontend.matcher_threshold").as_double());
    //Since we had to do a bunch of setup here -- it's out of sync. So lets ignore this, and just go to future image
    return;
  }
  cv::Mat inputImg = sensor_data->imageRaw();
  std::shared_ptr<sensor_msgs::msg::NavSatFix> gps_fix;
  if (sensor_handler_->enable_gps_recording_) {
    gps_fix = std::make_shared<sensor_msgs::msg::NavSatFix>(sensor_handler_->received_gps_queue_.back());
    sensor_handler_->received_gps_queue_.pop_back();
  }
  calcOdom.header.stamp = rtabmap_conversions::timestampToROS(sensor_data->stamp());
  const std::lock_guard<std::mutex> lock(odom_state_mutex);

  RCLCPP_DEBUG(get_logger(), "Processing new frame");

  if (keyframe_generation_ratio_threshold_ < 0.99f && keyframe_generation_ratio_threshold_ > 0.001f && 
      nb_local_keyframes_ > 0 && current_keyframe_ && current_OF_frame_ && odom_recovery_state != RECOVERY_FAILED) {
    rtabmap::RegistrationInfo reg_info;
    auto t = compute_flow(sensor_data, inputImg, reg_info);
    if (!t.isNull())
    {
      auto newPose = lastKFPose * t;

      RCLCPP_DEBUG(get_logger(), "New pose from internal tracking: %s", newPose.prettyPrint().c_str());
      float inliersRatio = (float) reg_info.inliers / (float) current_keyframe_->keypoints().size();
      if ( inliersRatio > keyframe_generation_ratio_threshold_)
      {
        RCLCPP_DEBUG(get_logger(), "Keeping OF FK due to high number of inliers from pervious KF %d %f", reg_info.inliers,
                        inliersRatio);
        publish_odom_update(newPose, reg_info.covariance);
        return;
      } else if (odom_recovery_state == ATTEMPTING_RECOVERY) {
        RCLCPP_DEBUG(get_logger(), "Keeping current KF due to recovery attempt");
        publish_odom_update(newPose, reg_info.covariance);
        return;
      }
      RCLCPP_DEBUG(get_logger(), "New OF frame -- not enought inliers %d %f", reg_info.inliers, inliersRatio);
    } else {
      RCLCPP_DEBUG(get_logger(), "Couldn't compute f2f transform with OF");
    }
  }


  if (compute_local_descriptors(sensor_data, inputImg)) {
    optical_matcher->updateBaseFrame(inputImg, sensor_data->keypoints());
    current_OF_frame_ = sensor_data;
    if (odom_recovery_state == RECOVERY_FAILED) {
      RCLCPP_DEBUG(get_logger(), "Recovery has failed -- purposefully sending a new KF");
    } else if (current_keyframe_) {
      rtabmap::RegistrationInfo reg_info;
      auto signatures = this->computeMatches(*current_keyframe_, *sensor_data);
      try {
        //RCLCPP_DEBUG(get_logger(), "Checking the matches computed transform");
        rtabmap::Transform t = intra_registration_->computeTransformation(
            *signatures.first, *signatures.second, rtabmap::Transform(), &reg_info);
        if ( !t.isNull()) {
          lastKFPose = lastKFPose * t;
          sensor_data->setGlobalPose(lastKFPose.clone(), reg_info.covariance);
          float inliersRatio = (float) reg_info.inliers / (float) current_keyframe_->keypoints().size();
          publish_odom_update(lastKFPose, reg_info.covariance);
          if (inliersRatio > keyframe_generation_ratio_threshold_ )
          {
            RCLCPP_DEBUG(get_logger(), "After feature matching -- frame still close to prev kf %d %f. Keeping prev KF", reg_info.inliers,
                            inliersRatio);
            return;
          } else if (odom_recovery_state == ATTEMPTING_RECOVERY) {
            RCLCPP_DEBUG(get_logger(), "Keeping current KF due to recovery attempt");
            return;
          } else {
            RCLCPP_DEBUG(get_logger(), "After feature matching -- frame not close to prev kf %d %f. Making new FK", reg_info.inliers,
                  inliersRatio);
          }
        } else {
          lastKFPose.setIdentity();
          if (odom_status != EXTERNAL) {
            RCLCPP_DEBUG(get_logger(), "Couldnt compute f2f transform with FM: inliers %d - ratio %f - matches %d. Waiting for loop closure reset", reg_info.inliers, reg_info.inliersRatio, reg_info.matches);
            odom_status = LOCAL_TRACKING;
          }
        } 
      }
      catch (std::exception &e)
      {
        RCLCPP_WARN(
            get_logger(),
            "Exception: Could not compute transform between keyframes: %s -- from words %lu words3 %lu : to words %lu words3 %lu",
            e.what(), signatures.first->getWords().size(), signatures.first->getWords3().size(), signatures.second->getWords().size(), signatures.second->getWords3().size());
        if (odom_status != EXTERNAL) odom_status = LOCAL_TRACKING;
      }
    } else if (odom_status != EXTERNAL) {
      odom_status = LOCAL_TRACKING;
    }
    
    current_keyframe_ = sensor_data;
    if (odom_status == GLOBAL_TRACKING || odom_status == EXTERNAL) {
      const std::lock_guard<std::mutex> map_lock(map_mutex);                 // Set keyframe ID
      sensor_data->setId(nb_local_keyframes_);
      local_descriptors_map_.insert({sensor_data->id(), sensor_data});
      nb_local_keyframes_++;
    } else {
      odom_recovery_state = ATTEMPTING_RECOVERY;
      sensor_data->setId(recoveryFrameId++);
    }

    if (odom_status == GLOBAL_TRACKING) {
      odom = std::make_shared<const nav_msgs::msg::Odometry>(calcOdom);
    }
    
    send_keyframe(std::make_pair(sensor_data, 
      (odom_status == GLOBAL_TRACKING || odom_status == EXTERNAL)? odom : nullptr), 
      sensor_handler_->enable_gps_recording_? gps_fix.get() : nullptr);
  } else {
    current_keyframe_ = nullptr;
    lastKFPose.setIdentity();
    odom_status = FAILURE;
  }
  clear_sensor_data(sensor_data);
}

void MapManager::sensor_data_to_rgbd_msg(
    const std::shared_ptr<rtabmap::SensorData> sensor_data,
    rtabmap_msgs::msg::SensorData &msg_data, bool baselinkFrame)
{
  rtabmap_conversions::sensorDataToROS(*sensor_data, msg_data);
  if (baselinkFrame) {
    rtabmap_conversions::points3fToROS(sensor_data->keypoints3D(), msg_data.points);
  }
}

void MapManager::local_descriptors_request(
    cslam_common_interfaces::msg::LocalDescriptorsRequest::
        ConstSharedPtr request)
{
  PROFILE_ME;
  // Fill msg
  auto msg = std::make_unique<cslam_common_interfaces::msg::LocalImageDescriptors>();
  std::shared_ptr<rtabmap::SensorData> sensorData;
  {
    const std::lock_guard<std::mutex> lock(map_mutex);
    sensorData = local_descriptors_map_.at(request->keyframe_id);
  }
  sensor_data_to_rgbd_msg(sensorData, msg->data);
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

void MapManager::recover_odom_pose(cslam_common_interfaces::msg::InterRobotMatches::ConstSharedPtr matches) {
  for(const auto match : matches->matches) {
    std::shared_ptr<rtabmap::SensorData> currentKF, matching_kf;
    {
      const std::lock_guard<std::mutex> odom_state_lock(odom_state_mutex);
      const std::lock_guard<std::mutex> map_lock(map_mutex);
      if (odom_status != LOCAL_TRACKING || current_keyframe_->id() != (int) match.robot1_keyframe_id) return;
      currentKF = current_keyframe_;
      matching_kf = local_descriptors_map_.at(match.robot0_keyframe_id);
    }
    auto signatures = this->computeMatches(*matching_kf, *currentKF);
    rtabmap::RegistrationInfo reg_info;
    rtabmap::Transform t = intra_registration_->computeTransformation(
                *signatures.first, *signatures.second, rtabmap::Transform(), &reg_info);
    const std::lock_guard<std::mutex> lock(odom_state_mutex);
    if (!t.isNull() && current_keyframe_ && current_keyframe_->id() == (int) match.robot1_keyframe_id) {
      //TODO be smarter about this -- we should be able to reuse the new map we're constructing
      // t.normalizeRotation();
      auto fromPose = matching_kf->globalPose();
      RCLCPP_DEBUG(get_logger(), "TF from reovery loop closure: %s * %s",fromPose.prettyPrint().c_str(), t.prettyPrint().c_str());
      auto newKFPose = fromPose.isNull()? t : fromPose * t;
      {
        const std::lock_guard<std::mutex> map_lock(map_mutex); 
        RCLCPP_DEBUG(get_logger(), "Tracking was lost -- resetting pose based on loop closure: %s", newKFPose.prettyPrint().c_str());
        lastKFPose = newKFPose;
        odom_status = GLOBAL_TRACKING;
        odom_recovery_state = RECOVERED;
        current_keyframe_->setGlobalPose(newKFPose, reg_info.covariance);
        current_keyframe_->setId(nb_local_keyframes_);
        local_descriptors_map_.insert({current_keyframe_->id(), current_keyframe_});
        nb_local_keyframes_++;
      }
      reg_info.covariance.reshape(1,1).copyTo(calcOdom.pose.covariance);
      rtabmap_conversions::transformToPoseMsg(newKFPose, calcOdom.pose.pose);
      calcOdom.header.stamp = rtabmap_conversions::timestampToROS(current_keyframe_->stamp());
      auto addKFMsg = std::make_unique<cslam_common_interfaces::msg::LocalKeyframeMatch>();
      addKFMsg->keyframe0_id = match.robot1_keyframe_id;
      addKFMsg->keyframe1_id = current_keyframe_->id();
      add_recovered_publisher_->publish(std::move(addKFMsg));

      auto odom_msg = std::make_unique<cslam_common_interfaces::msg::KeyframeOdom>();
      odom_msg->id = current_keyframe_->id();
      odom_msg->odom = calcOdom;
      keyframe_odom_publisher_->publish(std::move(odom_msg));
      return;
    } else {
      RCLCPP_DEBUG(get_logger(), "Could not re-align tracking based on new KF -- attempting next");
    }
  }
  const std::lock_guard<std::mutex> lock(odom_state_mutex);
  RCLCPP_DEBUG(get_logger(), "Could not align to any previous keyframes");
  odom_recovery_state = RECOVERY_FAILED;
}

void MapManager::receive_local_keyframe_match(
    cslam_common_interfaces::msg::LocalKeyframeMatch::ConstSharedPtr
        msg)
{
    PROFILE_ME;
    workerPool.enqueue([this, msg]() -> void {
      try {
        std::shared_ptr<rtabmap::SensorData> keyframe0;
        std::shared_ptr<rtabmap::SensorData> keyframe1;
        {
          const std::lock_guard<std::mutex> lock(map_mutex);
          keyframe0 = local_descriptors_map_.at(msg->keyframe0_id);
          keyframe1 = local_descriptors_map_.at(msg->keyframe1_id);
        }
        auto signatures = this->computeMatches(*keyframe0, *keyframe1);
        if (signatures.first == nullptr || signatures.second == nullptr) {
          auto lc = std::make_unique<cslam_common_interfaces::msg::IntraRobotLoopClosure>();
          lc->keyframe0_id = msg->keyframe0_id;
          lc->keyframe1_id = msg->keyframe1_id;
          lc->success = false;
          intra_robot_loop_closure_publisher_->publish(std::move(lc));
        } else {
          workerPool.enqueue([this, signatures, keyframe0, msg]() -> void
          {   PROFILE_ME_AS("Intra Registration");
              rtabmap::RegistrationInfo reg_info;

              rtabmap::Transform t = intra_registration_->computeTransformation(
                  *signatures.first, *signatures.second, rtabmap::Transform(), &reg_info);
              auto lc = std::make_unique<cslam_common_interfaces::msg::IntraRobotLoopClosure>();
              lc->keyframe0_id = msg->keyframe0_id;
              lc->keyframe1_id = msg->keyframe1_id;
              lc->success = false;
              if (!t.isNull())
              {
                // t.normalizeRotation();
                if (odom_status != EXTERNAL && (int) msg->keyframe1_id == current_keyframe_->id()) {
                  const std::lock_guard<std::mutex> pose_lock(odom_state_mutex);
                  lastKFPose = keyframe0->globalPose().isNull()? t : keyframe0->globalPose() * t;
                }
                lc->success = true;
                reg_info.covariance.reshape(1,1).copyTo(lc->pose.covariance);
                rtabmap_conversions::transformToPoseMsg(t, lc->pose.pose);
              }
              else
              {
                RCLCPP_DEBUG(
                    get_logger(),
                    "Intra-robot loop closure failed - could not compute transformation between (%d,%d) : %s",
                    lc->keyframe0_id, lc->keyframe1_id,
                    reg_info.rejectedMsg.c_str());
              }
              intra_robot_loop_closure_publisher_->publish(std::move(lc));
              
            });
        }
      }
      catch (std::exception &e)
      {
        RCLCPP_WARN(
            get_logger(),
            "Exception: Could not compute local transformation between %d and %d: %s",
            msg->keyframe0_id, msg->keyframe1_id,
            e.what());
      }
  });
}

void MapManager::local_descriptors_msg_to_sensor_data(
    const std::shared_ptr<
        cslam_common_interfaces::msg::LocalImageDescriptors>
        msg,
    rtabmap::SensorData &sensor_data)
{
  sensor_data = rtabmap_conversions::sensorDataFromROS(msg->data);
}

void MapManager::receive_local_image_descriptors(
    const std::shared_ptr<
        cslam_common_interfaces::msg::LocalImageDescriptors>
        msg)
{
  PROFILE_ME;
  std::deque<int> keyframe_ids;
  for (unsigned int i = 0; i < msg->matches_robot_id.size(); i++)
  {
    if (msg->matches_robot_id[i] == robot_id_)
    {
      keyframe_ids.push_back(msg->matches_keyframe_id[i]);
    }
  }
  auto to = std::make_shared<rtabmap::SensorData>();
  local_descriptors_msg_to_sensor_data(msg, *to);

  for (auto local_keyframe_id : keyframe_ids)
  {
    try
      {

        // Compute transformation
        //  Registration params
        rtabmap::RegistrationInfo reg_info;
        std::shared_ptr<rtabmap::SensorData> from;
        {
          const std::lock_guard<std::mutex> lock(map_mutex);
          from = local_descriptors_map_.at(local_keyframe_id);
        }

        workerPool.enqueue([this, local_keyframe_id, from, to, msg]() -> void {
            auto signatures = this->computeMatches(*from, *to);
            if (signatures.first == nullptr || signatures.second == nullptr) {
              auto lc = std::make_unique<cslam_common_interfaces::msg::InterRobotLoopClosure>();
              lc->robot0_id = robot_id_;
              lc->robot0_keyframe_id = local_keyframe_id;
              lc->robot1_id = msg->robot_id;
              lc->robot1_keyframe_id = msg->keyframe_id;
              lc->success = false;
              inter_robot_loop_closure_publisher_->publish(std::move(lc));
            } else {
              workerPool.enqueue([this, msg, signatures, local_keyframe_id]() -> void
                { 
                  PROFILE_ME_AS("Inter Registration");
                  auto lc = std::make_unique<cslam_common_interfaces::msg::InterRobotLoopClosure>();
                  lc->robot0_id = robot_id_;
                  lc->robot0_keyframe_id = local_keyframe_id;
                  lc->robot1_id = msg->robot_id;
                  lc->robot1_keyframe_id = msg->keyframe_id;
                  lc->success = false;
                  rtabmap::RegistrationInfo reg_info;

                  rtabmap::Transform t = inter_registration_->computeTransformation(
                      *signatures.first, *signatures.second, rtabmap::Transform(), &reg_info);
                
                  if (!t.isNull())
                  {
                    lc->success = true;
                    reg_info.covariance.reshape(1,1).copyTo(lc->pose.covariance);
                    rtabmap_conversions::transformToPoseMsg(t, lc->pose.pose);
                  }
                  else
                  {
                    RCLCPP_INFO(
                        get_logger(),
                        "Inter-robot loop closure failed between (%d,%d) and (%d,%d): %s \n Mean Distance: %f, total time: %f",
                        lc->robot0_id, lc->robot0_keyframe_id, lc->robot1_id, lc->robot1_keyframe_id,
                        reg_info.rejectedMsg.c_str(), reg_info.inliersMeanDistance, reg_info.totalTime);

                  }
                  inter_robot_loop_closure_publisher_->publish(std::move(lc));
                });
            }
        });

      }
      catch (std::exception &e)
      {
        RCLCPP_WARN(
            get_logger(),
            "Exception: Could not compute transformation between (%d,%d) and (%d,%d): %s",
            robot_id_, local_keyframe_id, msg->robot_id, msg->keyframe_id,
            e.what());
      }

  }
}

void MapManager::send_keyframe(const std::pair<std::shared_ptr<rtabmap::SensorData>, std::shared_ptr<const nav_msgs::msg::Odometry>> &keypoints_data, const sensor_msgs::msg::NavSatFix * gps_data)
{
  cv::Mat img;
  if (global_image_topic_.length() > 0) {
    keypoints_data.first->uncompressDataConst(0, 0, 0, &img);
  } else {
    keypoints_data.first->uncompressDataConst(&img, 0);
  }

  // Image message
  std_msgs::msg::Header header;
  if (keypoints_data.second)
    header.stamp = keypoints_data.second->header.stamp;
  else
    header.stamp = now();

  cv_bridge::CvImage image_bridge = cv_bridge::CvImage(header, img.channels() > 1? "bgr8":"mono8",img);
  auto keyframe_msg = std::make_unique<cslam_common_interfaces::msg::KeyframeRGB>();
  image_bridge.toImageMsg(keyframe_msg->image);
  keyframe_msg->image.header.frame_id = (odom_status == GLOBAL_TRACKING || odom_status == EXTERNAL)? "global" : "local";
  keyframe_msg->id = keypoints_data.first->id();
  keyframe_data_publisher_->publish(std::move(keyframe_msg));

  if (odom_status == GLOBAL_TRACKING || odom_status == EXTERNAL) {
    auto odom_msg = std::make_unique<cslam_common_interfaces::msg::KeyframeOdom>();
    odom_msg->id = keypoints_data.first->id();
    odom_msg->odom = *keypoints_data.second;
    if(gps_data != nullptr) 
      odom_msg->gps = *gps_data;

    keyframe_odom_publisher_->publish(std::move(odom_msg));

    if (enable_visualization_)
    {
      send_visualization(keypoints_data);
    }
  }

  // if(base_frame_id_.length() > 0 && hasTransform_) {

  //     tf2::Transform poseTf;
  //     tf2::fromMsg(odom_msg->odom.pose.pose, poseTf);
  //     auto out = poseTf * base_transform_;
  //     tf2::toMsg(out, odom_msg->odom.pose.pose);
  // }

}

void MapManager::send_visualization(const std::pair<std::shared_ptr<rtabmap::SensorData>, std::shared_ptr<const nav_msgs::msg::Odometry>> &keypoints_data)
{
  send_visualization_keypoints(keypoints_data);
  if (sensor_type == "rgbd") send_visualization_pointcloud(keypoints_data.first);
}

void MapManager::clear_sensor_data(std::shared_ptr<rtabmap::SensorData> sensor_data)
{
  // Clear costly data
  sensor_data->clearCompressedData();
  sensor_data->clearRawData();
}

void MapManager::send_visualization_keypoints(const std::pair<std::shared_ptr<rtabmap::SensorData>, std::shared_ptr<const nav_msgs::msg::Odometry>> &keypoints_data)
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

sensor_msgs::msg::PointCloud2 MapManager::visualization_pointcloud_voxel_subsampling(
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

void MapManager::send_visualization_pointcloud(const std::shared_ptr<rtabmap::SensorData> & sensor_data)
{
  cslam_common_interfaces::msg::VizPointCloud keyframe_pointcloud_msg;
  keyframe_pointcloud_msg.robot_id = robot_id_;
  keyframe_pointcloud_msg.keyframe_id = sensor_data->id();
  std_msgs::msg::Header header;
  header.stamp = now();
  header.frame_id = MAP_FRAME_ID(robot_id_);
  auto pointcloud_msg = create_colored_pointcloud(sensor_data, header);

  if (visualization_voxel_size_ > 0.0)
  {
    pointcloud_msg = visualization_pointcloud_voxel_subsampling(pointcloud_msg);
  }

  keyframe_pointcloud_msg.pointcloud = pointcloud_msg;
  keyframe_pointcloud_publisher_->publish(keyframe_pointcloud_msg);
}

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(cslam::MapManager)