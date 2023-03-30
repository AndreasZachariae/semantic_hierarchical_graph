import os
from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
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

    return LaunchDescription([
        Node(
            package='shg',
            executable='graph_node',
            name='graph_node',
            output='screen',
            parameters=[config],
            emulate_tty=True,
        )
    ])
