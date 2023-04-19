#!/usr/bin/env python3
import os
from time import time

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix
from nav_msgs.srv import GetPlan, GetMap
from nav_msgs.msg import OccupancyGrid
from map_msgs.srv import SaveMap
from geometry_msgs.msg import PoseStamped, Pose, Point, PointStamped
import tf_transformations

from semantic_hierarchical_graph.planners.shg_planner import SHGPlanner
from semantic_hierarchical_graph.types.exceptions import SHGPlannerError
from semantic_hierarchical_graph.types.position import Position


class GraphNode(Node):

    def __init__(self):
        super().__init__('graph_node')  # type: ignore

        self.graph_name = self.declare_parameter("graph_name", "graph").get_parameter_value().string_value
        self.initial_floor = self.declare_parameter("initial_floor", "ryu").get_parameter_value().string_value
        src_prefix = os.path.join(get_package_prefix('shg'), '..', '..', 'src', 'semantic_hierarchical_graph')

        self.get_logger().info("Graph name: " + str(self.graph_name) + ", initial floor: " + str(self.initial_floor))

        self.shg_planner = SHGPlanner(src_prefix + "/data/graphs/" + self.graph_name, "graph.pickle", False)

        # path_dict, distance = shg_planner.plan(["ryu", "room_8", "(1418, 90)"], ["hou2", "room_17", "(186, 505)"])
        # ryu_path = shg_planner.get_path_on_floor(["ryu"], key="name")
        # hou2_path = shg_planner.get_path_on_floor(["hou2"], key="name")
        # print(len(ryu_path))
        # print(len(hou2_path))

        self.plan_srv = self.create_service(GetPlan, 'shg/plan_path', self.plan_path_callback)
        self.map_client = self.create_client(GetMap, 'map_server/map')
        self.point_sub = self.create_subscription(PointStamped, 'clicked_point', self.clicked_point_callback, 10)
        self.goal_floor_srv = self.create_service(SaveMap, 'shg/goal_floor', self.goal_floor_callback)

        self.current_map = self.get_initial_map()
        self.current_floor_name = self.initial_floor
        self.goal_floor_name = None

        self.get_logger().info("Started graph_node")

    def get_initial_map(self):
        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('map_server/map service not available, waiting again...')

        future = self.map_client.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, future)

        initial_map: OccupancyGrid = future.result().map  # type: ignore
        return initial_map

    def goal_floor_callback(self, request: SaveMap.Request, response: SaveMap.Response) -> SaveMap.Response:
        self.goal_floor_name = request.filename
        self.get_logger().info("Goal floor: " + str(self.goal_floor_name))
        return response

    def clicked_point_callback(self, msg: PointStamped):
        self.get_logger().info("Clicked point: " + str(msg.point))
        pose = Pose()
        pose.position = msg.point
        pose.orientation.w = 1.0
        graph_hierarchy = self.transform_map_pos_to_graph_hierarchy(pose)  # msg.pose
        self.get_logger().info("Graph hierarchy: " + str(graph_hierarchy))

    def transform_map_pos_to_graph_hierarchy(self, pose: Pose):
        x = pose.position.x - self.current_map.info.origin.position.x
        y = pose.position.y - self.current_map.info.origin.position.y
        x = int(x / self.current_map.info.resolution)
        y = self.current_map.info.height - int(y / self.current_map.info.resolution)
        room = self.shg_planner.graph._get_child(self.current_floor_name).watershed[y, x]

        if room == 0 or room == 1:
            raise SHGPlannerError("Point is not in a valid room")

        room_str = "room_"+str(room)
        hierarchy = [self.current_floor_name, room_str, (x, y)]
        return hierarchy

    def transform_pixel_pos_to_map(self, position: Position):
        x = position.x * self.current_map.info.resolution + self.current_map.info.origin.position.x
        y = self.current_map.info.height - position.y
        y = y * self.current_map.info.resolution + self.current_map.info.origin.position.y
        return x, y, position.rz

    def plan_path_callback(self, request: GetPlan.Request, response: GetPlan.Response) -> GetPlan.Response:
        start_time = time()
        start_pose = request.start.pose
        goal_pose = request.goal.pose
        self.get_logger().info("Plan from (" + str(round(start_pose.position.x, 2)) + ", " + str(round(start_pose.position.y, 2)) +
                               ") to (" + str(round(goal_pose.position.x, 2)) + ", " + str(round(goal_pose.position.y, 2)) + ")")

        start_hierarchy = self.transform_map_pos_to_graph_hierarchy(start_pose)
        goal_hierarchy = self.transform_map_pos_to_graph_hierarchy(goal_pose)

        path_dict, distance = self.shg_planner.plan(start_hierarchy, goal_hierarchy)
        path = self.shg_planner.get_path_on_floor([self.current_floor_name], key="position")

        for pos in path:
            if pos == path[-1]:
                continue
            x, y, rz = self.transform_pixel_pos_to_map(pos)
            q = tf_transformations.quaternion_about_axis(rz, (0, 0, 1))
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            response.plan.poses.append(pose)  # type: ignore

        response.plan.header.frame_id = "map"
        response.plan.header.stamp = self.get_clock().now().to_msg()
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
