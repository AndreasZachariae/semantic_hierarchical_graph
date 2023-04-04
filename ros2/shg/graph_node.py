#!/usr/bin/env python3
import os
from time import time

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped

from semantic_hierarchical_graph.planners.shg_planner import SHGPlanner


class GraphNode(Node):

    def __init__(self):
        super().__init__('graph_node')  # type: ignore

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

        self.srv = self.create_service(GetPlan, 'shg/plan_path', self.plan_path_callback)

        self.get_logger().info("Started graph_node")

    def plan_path_callback(self, request: GetPlan.Request, response: GetPlan.Response) -> GetPlan.Response:
        start_time = time()
        start_pos = request.start.pose.position
        goal_pos = request.goal.pose.position
        self.get_logger().info("Plan from (" + str(round(start_pos.x, 2)) + ", " + str(round(start_pos.y, 2)) +
                               ") to (" + str(round(goal_pos.x, 2)) + ", " + str(round(goal_pos.y, 2)) + ")")

        path = [[-1.0, -0.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]
        for pos in path:
            pose = PoseStamped()
            pose.pose.position.x = pos[0]
            pose.pose.position.y = pos[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            response.plan.poses.append(pose)  # type: ignore

        response.plan.header.frame_id = "map"
        response.plan.header.stamp = self.get_clock().now().to_msg()
        # response.plan.poses.append(request.start)
        response.plan.poses.append(request.goal)  # type: ignore
        planning_time = time() - start_time
        self.get_logger().info("Planning time: " + str(round(planning_time, 6)))
        return response


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
