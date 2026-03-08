import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess,IncludeLaunchDescription,DeclareLaunchArgument
import launch

################### user configure parameters for ros2 start ###################
xfer_format   = 1    # 0-Pointcloud2(PointXYZRTL), 1-customized pointcloud format
multi_topic   = 0    # 0-All LiDARs share the same topic, 1-One LiDAR one topic
data_src      = 0    # 0-lidar, others-Invalid data src
publish_freq  = 10.0 # freqency of publish, 5.0, 10.0, 20.0, 50.0, etc.
output_type   = 0
frame_id      = 'livox_frame'
lvx_file_path = '/home/livox/livox_test.lvx'
cmdline_bd_code = 'livox0000000001'
pubvelodyne_pointcloud=True
raw_topic_suffix='/pc'
cur_path = os.path.split(os.path.realpath(__file__))[0] + '/'
cur_config_path = cur_path + '../config'
rviz_config_path = os.path.join(cur_config_path, 'display_point_cloud_ROS2.rviz')
user_config_path = os.path.join(cur_config_path, 'MID360_config.json')
################### user configure parameters for ros2 end #####################

livox_ros2_params = [
    {"xfer_format": xfer_format},
    {"multi_topic": multi_topic},
    {"data_src": data_src},
    {"publish_freq": publish_freq},
    {"output_data_type": output_type},
    {"frame_id": frame_id},
    {"lvx_file_path": lvx_file_path},
    {"user_config_path": user_config_path},
    {"cmdline_input_bd_code": cmdline_bd_code}
]


def generate_launch_description():
    ld=LaunchDescription()
    ld.add_action(DeclareLaunchArgument("use_rviz", default_value='true'))
    if(pubvelodyne_pointcloud==False):
        livox_driver = Node(
            package='livox_ros_driver2',
            executable='livox_ros_driver2_node',
            name='livox_lidar_publisher',
            output='screen',
            parameters=livox_ros2_params
            )
        ld.add_action(livox_driver)
    else:
        
        ld.add_action(DeclareLaunchArgument("sub_topic", default_value='livox/lidar'))
        ld.add_action(DeclareLaunchArgument("pub_topic", default_value='livox/lidar'+raw_topic_suffix))
        ld.add_action(DeclareLaunchArgument("pub_debug",default_value='false'))
        livox_driver_node=ComposableNode(
            package='livox_ros_driver2',
            plugin='livox_ros::DriverNode',
            name='livox_lidar_publisher',
            parameters=livox_ros2_params,
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        livox_bridge_node=ComposableNode(
            package='perception',
            plugin='Mid360Bridge',
            name='mid360_bridge',
            parameters=[{'sub_topic':LaunchConfiguration('sub_topic')},{'pub_topic':LaunchConfiguration('pub_topic')},{'pub_debug':LaunchConfiguration('pub_debug')}],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        container=ComposableNodeContainer(
            name='livox_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[livox_driver_node,livox_bridge_node],
            output='screen',
            emulate_tty=True
            
        )
        ld.add_action(container)
    
    livox_rviz = Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['--display-config', rviz_config_path],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        )
    ld.add_action(livox_rviz)
    return ld