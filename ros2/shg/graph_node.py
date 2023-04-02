#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix

from semantic_hierarchical_graph.planners.shg_planner import SHGPlanner
import os


class GraphNode(Node):

    def __init__(self):
        super().__init__('graph_node')

        self.declare_parameter("graph_name", "graph")
        self.graph_name = self.get_parameter("graph_name").get_parameter_value().string_value
        self.get_logger().info("Graph name: " + str(self.graph_name))
        src_prefix = os.path.join(get_package_prefix('shg'), '..', '..', 'src', 'semantic_hierarchical_graph')

        shg_planner = SHGPlanner(src_prefix + "/data/graphs/" + self.graph_name, "graph.pickle", False)

        path_dict, distance = shg_planner.plan(["ryu", "room_8", "(1418, 90)"], ["hou2", "room_17", "(186, 505)"])
        ryu_path = shg_planner.get_path_on_floor(["ryu"], only_names=True)
        hou2_path = shg_planner.get_path_on_floor(["hou2"], only_names=True)
        print(len(ryu_path))
        print(len(hou2_path))

        self.get_logger().info("Started graph_node")


def main(args=None):

    rclpy.init(args=args)

    node = GraphNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
