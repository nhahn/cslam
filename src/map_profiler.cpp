#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include "cslam/front_end/map_manager.h"
#include "cslam/MMeter.h"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  // Create executor
  rclcpp::executors::SingleThreadedExecutor executor;

  // Create nodes
  auto mapManager = std::make_shared<cslam::MapManager>(rclcpp::NodeOptions());
  // Add nodes to executor
  executor.add_node(mapManager);
  std::thread profiler(
    std::bind(&rclcpp::executors::SingleThreadedExecutor::spin,
   &executor));
  profiler.join();
  rclcpp::shutdown();

  std::cout << std::fixed << std::setprecision(6) << MMeter::getGlobalTreePtr()->totalsByDurationStr() << std::endl;
  std::cout << *MMeter::getGlobalTreePtr() << std::endl;
    MMeter::getGlobalTreePtr()->outputBranchPercentagesToOStream(std::cout);
  return 0;
}