import os
from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch_ros.actions import Node
import yaml


def generate_launch_description():

    src_config_prefix = os.path.join(get_package_prefix('shg'), '..', '..', 'src',
                                     'semantic_hierarchical_graph', 'ros2', 'config')

    config = os.path.join(src_config_prefix, 'params.yaml')
    config_file = yaml.load(open(config), Loader=yaml.FullLoader)

    graph_path = config_file["graph_node"]["ros__parameters"]["graph_path"]
    graph_config_path = os.path.join(src_config_prefix, graph_path, 'graph.yaml')
    graph_config = yaml.load(open(graph_config_path), Loader=yaml.FullLoader)
    initial_map = graph_config["initial_map"]
    map_paths = {map_config["hierarchy"][-1]:
                 os.path.join(src_config_prefix, graph_path, map_config["yaml_path"])
                 for map_config in graph_config["maps"]}

    return LaunchDescription([
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_paths[initial_map]}],
            arguments=['--ros-args', '--log-level', 'info']),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
            parameters=[{'autostart': True},
                        {'node_names': ['map_server']}])
    ])
