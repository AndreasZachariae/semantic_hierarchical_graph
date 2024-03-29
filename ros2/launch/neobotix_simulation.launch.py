# Neobotix GmbH
# Author: Pradheep Padmanabhan

import launch
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir, LaunchConfiguration
from launch_ros.actions import Node
import os
from pathlib import Path

MY_NEO_ROBOT = os.environ['ROBOT_NAME']
MY_NEO_ENVIRONMENT = os.environ['MAP_NAME']


def generate_launch_description():
    planner_src_dir = os.path.join(get_package_prefix('shg'), '..', '..', 'src', 'semantic_hierarchical_graph', 'ros2')
    aws_hospital_dir = get_package_share_directory('aws_robomaker_hospital_world')
    default_world_path = os.path.join(aws_hospital_dir, 'worlds', MY_NEO_ENVIRONMENT + '.world')

    # default_world_path = os.path.join(get_package_share_directory(
    #     'neo_simulation2'), 'worlds', MY_NEO_ENVIRONMENT + '.world')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    use_robot_state_pub = LaunchConfiguration('use_robot_state_pub')

    urdf = os.path.join(get_package_share_directory('neo_simulation2'),
                        'robots/'+MY_NEO_ROBOT+'/', MY_NEO_ROBOT+'.urdf')

    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py', arguments=[
                        '-entity', MY_NEO_ROBOT, '-file', urdf, '-x', '2', '-y', '2'], output='screen')

    rviz_config_dir = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'rviz',
        'nav2_default_view.rviz')

    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[urdf])

    teleop = Node(package='teleop_twist_keyboard', executable="teleop_twist_keyboard",
                  output='screen',
                  prefix='xterm -e',
                  name='teleop')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': default_world_path,
            'verbose': 'true'
        }.items()
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # NAVIGATION

    map_dir = LaunchConfiguration(
        'map',
        default=os.path.join(aws_hospital_dir, 'maps', 'aws_robomaker_hospital.yaml'))

    param_file_name = 'navigation.yaml'
    # param_dir = LaunchConfiguration(
    #     'params_file',
    #     default=os.path.join(
    #         get_package_share_directory('neo_simulation2'),
    #         'configs/'+MY_NEO_ROBOT,
    #         param_file_name))
    param_dir = LaunchConfiguration(
        'params_file',
        default=os.path.join(planner_src_dir,
                             'config', MY_NEO_ROBOT + '_' + param_file_name))

    nav2_launch_file_dir = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=map_dir,
            description='Full path to map file to load'),

        DeclareLaunchArgument(
            'params_file',
            default_value=param_dir,
            description='Full path to param file to load'),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([nav2_launch_file_dir, '/bringup_launch.py']),
            launch_arguments={
                'map': map_dir,
                'use_sim_time': use_sim_time,
                'params_file': param_dir}.items(),
        ),
        spawn_entity,
        start_robot_state_publisher_cmd,
        # teleop,
        gazebo,
        rviz])
