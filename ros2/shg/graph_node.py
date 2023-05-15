#!/usr/bin/env python3
import os
from time import time, sleep
import yaml

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_prefix
import tf_transformations
from nav_msgs.srv import GetPlan, GetMap
from nav2_msgs.srv import LoadMap
from nav_msgs.msg import OccupancyGrid
from map_msgs.srv import SaveMap
from geometry_msgs.msg import PoseStamped, Pose, Point, PointStamped

# Services for map change
from std_msgs.msg import Empty
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.srv import ClearEntireCostmap
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


from semantic_hierarchical_graph.planners.shg_planner import SHGPlanner
from semantic_hierarchical_graph.types.exceptions import SHGPlannerError
from semantic_hierarchical_graph.types.position import Position


class GraphNode(Node):

    def __init__(self):
        super().__init__('graph_node')  # type: ignore

        self.interpolation_resolution = self.declare_parameter(
            "interpolation_resolution", 1).get_parameter_value().double_value
        graph_path = self.declare_parameter("graph_path", "graph").get_parameter_value().string_value
        graph_path = os.path.join(get_package_prefix('shg'), '..', '..', 'src',
                                  'semantic_hierarchical_graph', 'ros2', 'config', graph_path)
        graph_config_path = os.path.join(graph_path, 'graph.yaml')
        print(graph_config_path)
        graph_config = yaml.load(open(graph_config_path), Loader=yaml.FullLoader)
        self.graph_name = graph_config["graph_name"]
        self.initial_map = graph_config["initial_map"]
        self.floor_height = graph_config["floor_height"]
        self.map_paths = {map_config["hierarchy"][-1]:
                          os.path.join(graph_path, map_config["yaml_path"])
                          for map_config in graph_config["maps"]}

        self.get_logger().info("Graph name: " + str(self.graph_name) + ", initial floor: " + str(self.initial_map))

        self.shg_planner = SHGPlanner(graph_path, "graph.pickle", False)

        self.plan_srv = self.create_service(GetPlan, 'shg/plan_path', self.plan_path_callback)
        self.map_client = self.create_client(GetMap, 'map_server/map')
        self.point_sub = self.create_subscription(PointStamped, 'clicked_point', self.clicked_point_callback, 10)
        self.goal_floor_srv = self.create_service(SaveMap, 'shg/goal_floor', self.goal_floor_callback)

        self.current_map = self.get_initial_map()
        self.current_floor_name = self.initial_map
        self.goal_floor_name = None

        self.shg_planner.update_floor((self.current_map.info.origin.position.x, self.current_map.info.origin.position.y),
                                      self.current_map.info.resolution,
                                      (self.current_map.info.height, self.current_map.info.width),
                                      self.current_floor_name)

        # Services to change map and spawn simulation robot on new floor
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.change_map_srv = self.create_service(LoadMap, 'shg/change_map', self.change_map_callback)
        self.gazebo_calls_group = MutuallyExclusiveCallbackGroup()
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10, callback_group=self.gazebo_calls_group)
        self.clear_local_costmap_client = self.create_client(
            ClearEntireCostmap, "/local_costmap/clear_entirely_local_costmap", callback_group=self.gazebo_calls_group)
        self.clear_global_costmap_client = self.create_client(
            ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap", callback_group=self.gazebo_calls_group)
        self.set_entity_state_client = self.create_client(
            SetEntityState, "/gazebo/set_entity_state", callback_group=self.gazebo_calls_group)
        self.load_map_client = self.create_client(
            LoadMap, "/map_server/load_map", callback_group=self.gazebo_calls_group)

        while not (self.set_entity_state_client.wait_for_service(timeout_sec=1.0) and
                   self.load_map_client.wait_for_service(timeout_sec=1.0) and
                   self.clear_local_costmap_client.wait_for_service(timeout_sec=1.0) and
                   self.clear_global_costmap_client.wait_for_service(timeout_sec=1.0)):
            self.get_logger().info('Necessary gazebo services not available, waiting again...')

        # ros2 service call /gazebo/get_entity_state gazebo_msgs/srv/GetEntityState "{name: turtlebot3_waffle}"
        # ros2 service call /gazebo/set_entity_state gazebo_msgs/srv/SetEntityState "{state: {name: turtlebot3_waffle, pose: {position: {x: 3, y: 2}}}}"

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

    def change_map_callback(self, request: LoadMap.Request, response: LoadMap.Response) -> LoadMap.Response:
        map_name = request.map_url
        if map_name not in self.map_paths:
            self.get_logger().error("Map name not available, choose from: " + str(self.map_paths.keys()))
            return response
        self.get_logger().info("Changing map to: " + str(map_name))

        # Get current position
        try:
            t = self.tf_buffer.lookup_transform(
                "map",
                "base_footprint",
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(
                f'Could not transform {"map"} to {"base_footprint"}: {ex}')
            return response

        # Teleport robot to new floor
        new_position = SetEntityState.Request()
        new_position.state.name = "turtlebot3_waffle"
        new_position.state.pose.position.x = t.transform.translation.x
        new_position.state.pose.position.y = t.transform.translation.y
        new_position.state.pose.position.z = t.transform.translation.z + self.floor_height
        new_position.state.pose.orientation = t.transform.rotation

        res = self.set_entity_state_client.call(new_position)
        if not res.success:
            self.get_logger().error("Could not set new position")
            return response
        self.get_logger().info("New position set")

        # Load new map
        req = LoadMap.Request()
        req.map_url = self.map_paths[map_name]
        res = self.load_map_client.call(req)
        if not res:
            self.get_logger().error("Could not load map")
            return response
        self.get_logger().info("Map loaded")

        sleep(2)

        # Clear costmaps
        req = ClearEntireCostmap.Request()
        req.request = Empty()
        res = self.clear_local_costmap_client.call(req)
        if not res:
            self.get_logger().error("Could not clear local costmap")
            return response
        res = self.clear_global_costmap_client.call(req)
        if not res:
            self.get_logger().error("Could not clear global costmap")
            return response
        self.get_logger().info("Costmaps cleared")

        return response

    def clicked_point_callback(self, msg: PointStamped):
        self.get_logger().info("Clicked point: " + str(msg.point))
        x, y = Position.from_map_frame((msg.point.x, msg.point.y),
                                       (self.current_map.info.origin.position.x, self.current_map.info.origin.position.y),
                                       self.current_map.info.resolution,
                                       (self.current_map.info.height, self.current_map.info.width)).xy
        print(x, y)
        room = self.shg_planner.graph._get_child(self.current_floor_name).watershed[y, x]
        room_str = "room_"+str(room)
        self.get_logger().info("Graph hierarchy: " + str([self.current_floor_name, room_str, (x, y)]))

    def plan_path_callback(self, request: GetPlan.Request, response: GetPlan.Response) -> GetPlan.Response:
        start_time = time()
        start_pose = (request.start.pose.position.x, request.start.pose.position.y)
        goal_pose = (request.goal.pose.position.x, request.goal.pose.position.y)
        self.get_logger().info("Plan from (" + str(round(start_pose[0], 2)) + ", " + str(round(start_pose[1], 2)) +
                               ") to (" + str(round(goal_pose[0], 2)) + ", " + str(round(goal_pose[1], 2)) + ")")

        # TODO: Use correct floors for hierarchical planning
        start_floor = self.current_floor_name
        goal_floor = self.current_floor_name

        path_list, distance = self.shg_planner.plan_in_map_frame(start_pose, start_floor,
                                                                 goal_pose, goal_floor,
                                                                 self.interpolation_resolution)

        for x, y, rz in path_list:
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            if tuple([x, y, rz]) == path_list[-1]:
                pose.pose.orientation = request.goal.pose.orientation
            else:
                q = tf_transformations.quaternion_about_axis(rz, (0, 0, 1))
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]

            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            response.plan.poses.append(pose)  # type: ignore

        response.plan.header.frame_id = "map"
        response.plan.header.stamp = self.get_clock().now().to_msg()
        planning_time = time() - start_time
        self.get_logger().info("Path length: " + str(round(distance, 2)) + ", nodes:" + str(len(response.plan.poses)))
        self.get_logger().info("Planning time: " + str(round(planning_time, 6)))
        return response


def main(args=None):

    rclpy.init(args=args)

    node = GraphNode()
    executor = MultiThreadedExecutor()

    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
