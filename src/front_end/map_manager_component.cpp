#include <memory>
#include <tuple>
#include "rclcpp/rclcpp.hpp"
#include "cslam/front_end/map_manager.h"
#include <rtabmap/utilite/ULogger.h>
#include "cslam/MMeter.h"

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
      MapManagerComponent(rclcpp::NodeOptions ops) : Node("map_manager", ops.start_parameter_event_publisher(false).start_parameter_services(false))
      {

      }

  };
};

