cmake_minimum_required(VERSION 3.5)
project(cslam)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -g)
endif()

set(ONNXRuntime_ROOT_DIR "/opt/onnxruntime")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_msgs REQUIRED)
#find_package(eigen_conversions REQUIRED)
find_package(laser_geometry REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
#find_package(dynamic_reconfigure REQUIRED)
find_package(message_filters REQUIRED)
find_package(class_loader REQUIRED)
#find_package(rosgraph_msgs REQUIRED)
find_package(image_geometry REQUIRED)
#find_package(pluginlib REQUIRED)
# find_package(rtabmap_ros REQUIRED)
find_package(rtabmap_msgs REQUIRED)
find_package(rtabmap_conversions REQUIRED)
find_package(lightglue_onnx REQUIRED)
find_package(RTABMap REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find GTSAM
find_package(GTSAM CONFIG REQUIRED)

# kinetic issue, rtabmap now requires at least c++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
set(CMAKE_CXX_STANDARD 17)

set(ONNXRuntime_ROOT_DIR "/opt/onnxruntime")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindONNXRuntime.cmake)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  # ${rtabmap_ros_INCLUDE_DIRS}
  ${rtabmap_msgs_INCLUDE_DIRS}
  ${rtabmap_conversions_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# libraries
SET(Libraries
   pcl_conversions
   cv_bridge
   rclcpp
   rclcpp_components
   sensor_msgs
   std_msgs
   nav_msgs
   nav2_msgs
   geometry_msgs
   image_transport
   tf2
   tf2_eigen
   tf2_ros
   laser_geometry
   message_filters
   class_loader
   visualization_msgs
   image_geometry
   stereo_msgs
   diagnostic_msgs
   tf2_geometry_msgs
   pcl_msgs
   
  #  rtabmap_ros
   rtabmap_msgs
   rtabmap_conversions
)

find_package(cslam_common_interfaces REQUIRED)
set(node_plugins "")

add_library(pose_graph_manager_component SHARED
  src/back_end/decentralized_pgo.cpp
  src/back_end/pose_graph_manager_component.cpp
  src/back_end/utils/logger.cpp
  src/back_end/gtsam_utils.cpp
  src/back_end/utils/simulated_rendezvous.cpp)
ament_target_dependencies(pose_graph_manager_component 
  ${Libraries}
  cslam_common_interfaces)
target_link_libraries(pose_graph_manager_component gtsam)
target_include_directories(pose_graph_manager_component PUBLIC
  ${GTSAM_INCLUDE_DIR}
)
rclcpp_components_register_node(pose_graph_manager_component PLUGIN "cslam::PoseGraphManagerComponent" EXECUTABLE pose_graph_manager)
set(node_plugins "${node_plugins}cslam::PoseGraphManagerComponent;$<TARGET_FILE:pose_graph_manager_component>\n")

add_library(map_manager_component SHARED
  src/front_end/map_manager.cpp 
  src/front_end/map_manager_component.cpp 
  src/front_end/stereo_handler.cpp 
  src/front_end/rgbd_handler.cpp 
  #src/front_end/sensor_msg_utils.cpp
  src/front_end/visualization_utils.cpp
)
target_include_directories(map_manager_component PUBLIC
  ${RTABMap_INCLUDE_DIRS}
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
)
ament_target_dependencies(map_manager_component 
  ${Libraries} RTABMap cslam_common_interfaces lightglue_onnx)
rclcpp_components_register_node(map_manager_component PLUGIN "cslam::MapManagerComponent" EXECUTABLE map_manager)
set(node_plugins "${node_plugins}cslam::MapManagerComponent;$<TARGET_FILE:map_manager_component>\n")

cuda_add_library(global_descriptor_component SHARED
  src/front_end/global_descriptor_component.cpp
)
target_link_libraries(global_descriptor_component ${ONNXRuntime_LIBRARIES})
ament_target_dependencies(global_descriptor_component 
  ${Libraries}  
  cslam_common_interfaces lightglue_onnx)
target_include_directories(global_descriptor_component PUBLIC
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
)
rclcpp_components_register_node(global_descriptor_component PLUGIN "cslam::GlobalDescriptorComponent" EXECUTABLE global_descriptor)
set(node_plugins "${node_plugins}cslam::GlobalDescriptorComponent;$<TARGET_FILE:global_descriptor_component>\n")

# Only required when using messages built from the same package
# https://index.ros.org/doc/ros2/Tutorials/Rosidl-Tutorial/
get_default_rmw_implementation(rmw_implementation)
find_package("${rmw_implementation}" REQUIRED)
get_rmw_typesupport(typesupport_impls "${rmw_implementation}" LANGUAGE "cpp")

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
   DESTINATION include/${PROJECT_NAME}
   FILES_MATCHING PATTERN "*.h"
   PATTERN ".svn" EXCLUDE
)

install(DIRECTORY 
   models
   DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY 
   launch
   DESTINATION share/${PROJECT_NAME}/
)

install(TARGETS map_manager_component pose_graph_manager_component global_descriptor_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

file(GENERATE
OUTPUT
"${CMAKE_CURRENT_BINARY_DIR}/test_ament_index/$<CONFIG>/share/ament_index/resource_index/node_plugin/${PROJECT_NAME}"
CONTENT "${node_plugins}")

# Install nodes
install(
  TARGETS map_manager pose_graph_manager global_descriptor
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS 
cslam/loop_closure_detection_node.py 
cslam/lidar_handler_node.py 
DESTINATION lib/${PROJECT_NAME})

# Add tests
find_package(ament_cmake_pytest REQUIRED)
set(_cslam_python_tests
      tests/test_algebraic_connectivity.py
      tests/test_sparse_matching.py
      tests/test_broker.py
  )
foreach(_test_path ${_cslam_python_tests})
  get_filename_component(_test_name ${_test_path} NAME_WE)
  ament_add_pytest_test(${_test_name} ${_test_path}
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 120
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
endforeach()

ament_package()