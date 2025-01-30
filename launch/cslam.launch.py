import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ParameterFile, ComposableNode


def launch_setup(context, *args, **kwargs):
    loop_detection_node = Node(package='cslam',
                               executable='loop_closure_detection_node.py',
                               name='cslam_loop_closure_detection',
                               
                               parameters=[
                                   ParameterFile(LaunchConfiguration('base_params').perform(context), allow_substs=True),
                                   ParameterFile(LaunchConfiguration('robot_params').perform(context), allow_substs=True), {
                                       "robot_id": LaunchConfiguration('robot_id'),
                                       "max_nb_robots": LaunchConfiguration('max_nb_robots'),
                                        "tf_prefix": LaunchConfiguration('tf_prefix'),
                                   }
                               ],
                            #    prefix=['stdbuf -o L'],
                            #    output='screen',
                            #   prefix="pprofile -o cslam.pprofile",
                                arguments=['--ros-args','--log-level',LaunchConfiguration('log_level'),'--log-level','rcl:=INFO', '--log-level','rmw_zenoh_cpp:=FATAL'],
                               namespace=LaunchConfiguration('namespace'))

    pose_graph_manager_component = ComposableNode(
                                package='cslam',
                                namespace=LaunchConfiguration('namespace'),
                                plugin='cslam::PoseGraphManagerComponent',
                                name=f"pose_graph_manager",
                                parameters=[
                                ParameterFile(LaunchConfiguration('base_params').perform(context), allow_substs=True),
                                ParameterFile(LaunchConfiguration('robot_params').perform(context), allow_substs=True), {
                                        "robot_id": LaunchConfiguration('robot_id'),
                                        "max_nb_robots": LaunchConfiguration('max_nb_robots'),
                                        "evaluation.enable_simulated_rendezvous": LaunchConfiguration('enable_simulated_rendezvous'),
                                        "evaluation.rendezvous_schedule_file": LaunchConfiguration('rendezvous_schedule_file'),
                                        "tf_prefix": LaunchConfiguration('tf_prefix'),
                                    }
                                ],
                                extra_arguments=[{'use_intra_process_comms': True}]
                            )
    map_manager = ComposableNode(
                            package='cslam',
                            plugin='cslam::MapManager',
                            namespace=LaunchConfiguration('namespace'),
                            name=f"map_manager",
                            parameters=[
                            ParameterFile(LaunchConfiguration('base_params').perform(context), allow_substs=True),
                            ParameterFile(LaunchConfiguration('robot_params').perform(context), allow_substs=True), {
                                    "robot_id": LaunchConfiguration('robot_id'),
                                    "max_nb_robots": LaunchConfiguration('max_nb_robots'),
                                    "tf_prefix": LaunchConfiguration('tf_prefix'),
                                }
                            ],
                            extra_arguments=[{'use_intra_process_comms': True}]
                        )

    global_descriptor_component = ComposableNode(
                                package='cslam',
                                namespace=LaunchConfiguration('namespace'),
                                plugin='cslam::GlobalDescriptorComponent',
                                name=f"global_descriptor",
                                parameters=[
                                ParameterFile(LaunchConfiguration('base_params').perform(context), allow_substs=True),
                                ParameterFile(LaunchConfiguration('robot_params').perform(context), allow_substs=True), {
                                        "robot_id": LaunchConfiguration('robot_id'),
                                        "max_nb_robots": LaunchConfiguration('max_nb_robots'),
                                        "evaluation.enable_simulated_rendezvous": LaunchConfiguration('enable_simulated_rendezvous'),
                                        "evaluation.rendezvous_schedule_file": LaunchConfiguration('rendezvous_schedule_file'),
                                        "tf_prefix": LaunchConfiguration('tf_prefix'),
                                    }
                                ],
                                extra_arguments=[{'use_intra_process_comms': True}]
                            )

    return [
        loop_detection_node,
        ComposableNodeContainer(
                namespace=LaunchConfiguration('namespace'),
                name='map_container',
                package='rclcpp_components',
                executable='component_container_mt',
                arguments=['--ros-args','--log-level',LaunchConfiguration('log_level'),'--log-level','rcl:=INFO', '--log-level','rmw_zenoh_cpp:=FATAL'],
                composable_node_descriptions=[global_descriptor_component, map_manager],
                prefix=['stdbuf -o L'],
                output='screen',
                #  prefix="gdbgui --args",
        ), 
        ComposableNodeContainer(
                namespace=LaunchConfiguration('namespace'),
                name='pose_container',
                package='rclcpp_components',
                executable='component_container_mt',
                arguments=['--ros-args','--log-level',LaunchConfiguration('log_level'),'--log-level','rcl:=INFO', '--log-level','rmw_zenoh_cpp:=FATAL'],
                composable_node_descriptions=[pose_graph_manager_component],
                prefix=['stdbuf -o L'],
                output='screen',
                #  prefix="gdbgui --args",
        ),
    ]


def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='', description=''),
        DeclareLaunchArgument('tf_prefix', default_value=PythonExpression(['("', LaunchConfiguration('namespace'), '".strip("/") + "/").lstrip("/")'])),
        DeclareLaunchArgument('robot_id', default_value='0', description=''),
        DeclareLaunchArgument('max_nb_robots', default_value='2', description=''),
        DeclareLaunchArgument('config_path', default_value='/config/', description=''),
        DeclareLaunchArgument('base_config', default_value='cslam_shared.yaml', description=''),
        DeclareLaunchArgument('robot_config', default_value='hl2_stereo.yaml', description=''),
        DeclareLaunchArgument('base_params',
                              default_value=[
                                  LaunchConfiguration('config_path'),
                                  LaunchConfiguration('base_config')
                              ],
                              description=''),
        DeclareLaunchArgument('robot_params',
                        default_value=[
                            LaunchConfiguration('config_path'),
                            LaunchConfiguration('robot_config')
                        ],
                        description=''),
        DeclareLaunchArgument(
            'launch_prefix_cslam',
            default_value='',
            description=
            'For debugging purpose, it fills prefix tag of the nodes, e.g., "xterm -e gdb -ex run --args"'
        ),
        DeclareLaunchArgument('enable_simulated_rendezvous', default_value='false', description=''),
        DeclareLaunchArgument('rendezvous_schedule_file', default_value='', description=''),
        DeclareLaunchArgument('log_level', default_value='info', description=''),
        OpaqueFunction(function=launch_setup)
    ])

