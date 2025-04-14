#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include "cslam/front_end/map_manager.h"
#include "cslam/profiler.h"


int main(int argc, char * argv[])
{
  PROFILE_ME;
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
	PROFILE_END; // Early profile stop to finalize results for main

	// std::cout << profiler::getInstance() << std::endl;
	profiler::getInstance().print(std::cout, 60);
  return 0;
}