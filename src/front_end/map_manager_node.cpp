#include "cslam/front_end/map_manager.h"
#include <rtabmap/utilite/ULogger.h>

using namespace cslam;

/**
 * @brief Node to manage the sensor data and registration
 *
 * @param argc
 * @param argv
 * @return int
 */

std::map<std::string, ULogger::Level> rtabmapLogLevel =
{
    { "debug", ULogger::kDebug },
    { "info", ULogger::kInfo },
    { "warning", ULogger::kWarning },
    { "error", ULogger::kError },
    { "fatal", ULogger::kFatal }
};

int main(int argc, char **argv) {

  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("map_manager");

  //Adjustable RTABMAP log level here
  node->declare_parameter<std::string>("rtabmap.log_level", "warning");
  ULogger::setType(ULogger::kTypeConsole);
  auto level = rtabmapLogLevel.find(node->get_parameter("rtabmap.log_level").as_string());
  ULogger::setLevel(level == rtabmapLogLevel.end()? ULogger::kWarning : level->second);

  node->declare_parameter<int>("frontend.pnp_min_inliers", 20);
  node->declare_parameter<int>("frontend.max_queue_size", 10);
  node->declare_parameter<int>("max_nb_robots", 1);
  node->declare_parameter<int>("robot_id", 0);
  node->declare_parameter<int>("frontend.map_manager_process_period_ms", 100);
  node->declare_parameter<std::string>("frontend.sensor_type", "stereo");
  node->declare_parameter<bool>("visualization.enable", false);
  node->declare_parameter<int>("visualization.publishing_period_ms", 0);
  node->declare_parameter<float>("visualization.voxel_size", 0.05);
  node->declare_parameter<float>("visualization.max_range", 2.0);
  node->declare_parameter<bool>("evaluation.enable_gps_recording", false);
  node->declare_parameter<std::string>("evaluation.gps_topic", "");

  std::string sensor_type;
  node->get_parameter("frontend.sensor_type", sensor_type);

  std::shared_ptr<IMapManager> handler;       
  if (sensor_type == "stereo") {
    handler = std::make_shared<MapManager<StereoHandler>>(node);
  } 
  else if (sensor_type == "rgbd") {
    handler = std::make_shared<MapManager<RGBDHandler>>(node);
  } 
  else {
    RCLCPP_ERROR(node->get_logger(), "Sensor type not supported: %s",
                 sensor_type.c_str());
    return -1;
  }

  rclcpp::spin(node);

  rclcpp::shutdown();

  return 0;
}
