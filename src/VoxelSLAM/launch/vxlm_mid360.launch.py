import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_voxel_slam = get_package_share_directory('voxel_slam')
    
    rviz_config_path = os.path.join(pkg_voxel_slam, 'rviz_cfg', 'back.rviz')
    default_config_path = os.path.join(pkg_voxel_slam, 'config', 'mid360.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Whether to start RViz'
        ),
        
        Node(
            package='voxel_slam',
            executable='voxelslam',
            name='voxelslam',
            output='screen',
            parameters=[default_config_path]
        ),
        
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            condition=IfCondition(LaunchConfiguration('rviz'))
        )
    ])
