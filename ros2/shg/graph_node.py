#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix

from semantic_hierarchical_graph.floor import Floor
from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.types.position import Position
import os


class GraphNode(Node):

    def __init__(self):
        super().__init__('graph_node')

        self.declare_parameter("parameter", 0)
        self.parameter_ = self.get_parameter("parameter").get_parameter_value().integer_value
        pkg_prefix = os.path.join(get_package_prefix('shg'), 'lib', 'shg')

        G = SHGraph(root_name="Benchmark", root_pos=Position(0, 0, 0))
        floor = Floor("ryu", G, Position(0, 0, 1), pkg_prefix + '/data/benchmark_maps/ryu.png',
                      pkg_prefix + "/config/ryu_params.yaml")
        G.add_child_by_node(floor)
        print(G.get_childs("name"))

        self.get_logger().info("parameter = " + str(self.parameter_))
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
