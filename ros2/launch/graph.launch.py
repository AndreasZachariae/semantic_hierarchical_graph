import os
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_prefix('shg'),
        '..',
        '..',
        'src',
        'semantic_hierarchical_graph',
        'ros2',
        'config',
        'params.yaml'
    )

    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            get_package_share_directory('shg') + '/launch/tb3_simulation_launch.py'))

    return LaunchDescription([
        # simulation_launch,
        Node(
            package='shg',
            executable='graph_node.py',
            name='graph_node',
            output='screen',
            parameters=[config],
            emulate_tty=True,
        )
    ])
