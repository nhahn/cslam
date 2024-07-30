#include <memory>
#include <tuple>
#include "rclcpp/rclcpp.hpp"
#include "cslam/front_end/map_manager.h"
#include <rtabmap/utilite/ULogger.h>

std::map<std::string, ULogger::Level> rtabmapLogLevel =
{
    { "debug", ULogger::kDebug },
    { "info", ULogger::kInfo },
    { "warning", ULogger::kWarning },
    { "error", ULogger::kError },
    { "fatal", ULogger::kFatal }
};


namespace cslam {
  class MapManagerComponent : public rclcpp::Node
  {
    public:
      std::shared_ptr<IMapManager> handler;       
      MapManagerComponent(rclcpp::NodeOptions ops) : Node("map_manager", ops)
      {
        declare_parameter<std::string>("rtabmap.log_level", "warning");
        ULogger::setType(ULogger::kTypeConsole);
        auto level = rtabmapLogLevel.find(get_parameter("rtabmap.log_level").as_string());
        ULogger::setLevel(level == rtabmapLogLevel.end()? ULogger::kWarning : level->second);

        declare_parameter<int>("frontend.pnp_min_inliers", 20);
        declare_parameter<float>("frontend.min_3d_keypoints", 100);
        declare_parameter<int>("frontend.inter_pnp_min_inliers", get_parameter("frontend.pnp_min_inliers").as_int());
        declare_parameter<int>("frontend.intra_pnp_min_inliers", get_parameter("frontend.pnp_min_inliers").as_int());
        declare_parameter<int>("frontend.max_queue_size", 10);
        declare_parameter<int>("max_nb_robots", 1);
        declare_parameter<int>("robot_id", 0);
        declare_parameter<int>("frontend.map_manager_process_period_ms", 100);
        declare_parameter<std::string>("frontend.sensor_type", "stereo");
        declare_parameter<bool>("visualization.enable", false);
        declare_parameter<int>("visualization.publishing_period_ms", 0);
        declare_parameter<float>("visualization.voxel_size", 0.05);
        declare_parameter<float>("visualization.max_range", 2.0);
        declare_parameter<bool>("evaluation.enable_gps_recording", false);
        declare_parameter<std::string>("evaluation.gps_topic", "");

        std::string sensor_type;
        get_parameter("frontend.sensor_type", sensor_type);
        if (sensor_type == "stereo") {
            handler = std::make_shared<MapManager<StereoHandler>>(this);
        } 
        else if (sensor_type == "rgbd") {
            handler = std::make_shared<MapManager<RGBDHandler>>(this);
        } 
        else {
            RCLCPP_ERROR(get_logger(), "Sensor type not supported: %s",
                        sensor_type.c_str());
        }
      }
  };
};

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(cslam::MapManagerComponent)