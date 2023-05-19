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
from shg_interfaces.srv import GetElevatorData


from semantic_hierarchical_graph.planners.shg_planner import SHGPlanner
from semantic_hierarchical_graph.types.exceptions import SHGPlannerError
from semantic_hierarchical_graph.types.position import Position, convert_map_pos_to_hierarchy
from semantic_hierarchical_graph.utils import path_to_list


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
        self.graph_config = yaml.load(open(graph_config_path), Loader=yaml.FullLoader)
        self.graph_name = self.graph_config["graph_name"]
        self.initial_map = self.graph_config["initial_map"]
        self.floor_height = self.graph_config["floor_height"]
        self.map_paths = {map_config["hierarchy"][-1]:
                          os.path.join(graph_path, map_config["yaml_path"])
                          for map_config in self.graph_config["maps"]}
        self.force_build_new_graph = self.declare_parameter(
            "force_build_new_graph", False).get_parameter_value().bool_value

        self.get_logger().info("Graph name: " + str(self.graph_name) + ", initial floor: " + str(self.initial_map))

        self.shg_planner = SHGPlanner(graph_path, "graph.pickle", self.force_build_new_graph)

        self.plan_srv = self.create_service(GetPlan, 'shg/plan_path', self.plan_path_callback)
        self.map_client = self.create_client(GetMap, 'map_server/map')
        self.point_sub = self.create_subscription(PointStamped, 'clicked_point', self.clicked_point_callback, 10)
        self.get_elevator_data_srv = self.create_service(
            GetElevatorData, 'shg/get_elevator_data', self.get_elevator_data_callback)

        self.current_map = self.get_initial_map()
        self.current_floor_name = self.initial_map

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

    def get_elevator_data_callback(self, request: GetElevatorData.Request, response: GetElevatorData.Response) -> GetElevatorData.Response:
        # get next floor
        floor_list = path_to_list(self.shg_planner.path, [], True)
        next_index = floor_list.index(self.current_floor_name) + 1
        # if next_index >= len(floor_list):
        #     self.get_logger().info("No more floors to visit")
        #     return response
        # next_floor = floor_list[next_index]
        next_floor = self.current_floor_name
        response.next_floor = next_floor
        print("Next floor: " + str(next_floor))

        # Get elevator direction
        current_pos_z = self.shg_planner.graph.get_child_by_hierarchy([self.current_floor_name]).pos_abs.z
        next_pos_z = self.shg_planner.graph.get_child_by_hierarchy([next_floor]).pos_abs.z
        direction = "up" if next_pos_z > current_pos_z else "down"
        response.direction = direction
        print("Direction: " + str(direction))

        # Get marker id
        for map_config in self.graph_config["maps"]:
            if next_floor in map_config["hierarchy"]:
                marker_id = map_config["marker_id"]
                break
        else:
            self.get_logger().error("Floor " + str(next_floor) + " not found in graph config")
            return response
        response.marker_id = marker_id
        print("Marker id: " + str(marker_id))

        # Get elevator parameters
        current_path_list = self.shg_planner.get_path_on_floor([next_floor], "node", None)
        # import semantic_hierarchical_graph.utils as utils
        # print("current_path_list", utils.map_names_to_nodes(current_path_list))
        print(current_path_list[-1].data_dict)
        if current_path_list[-1].data_dict:
            orientation_angle = current_path_list[-1].data_dict["orientation_angle"]
            call_button = current_path_list[-1].data_dict["call_button"]
            selection_panel = current_path_list[-1].data_dict["selection_panel"]
            response.orientation_angle = orientation_angle
            response.call_button.x = call_button[0]
            response.call_button.y = call_button[1]
            response.call_button.z = call_button[2]
            response.selection_panel.x = selection_panel[0]
            response.selection_panel.y = selection_panel[1]
            response.selection_panel.z = selection_panel[2]
            print("orientation_angle", orientation_angle)
            print("call_button", call_button)
            print("selection_panel", selection_panel)

        # Get next floor start pose
        path_list = self.shg_planner.get_path_on_floor([next_floor], "position", None)
        if len(path_list) == 0:
            self.get_logger().error("No path found on selected floor, call 'shg/plan_path' service first " + str(next_floor))
            return response

        next_floor_start: Position = path_list[0]
        start_in_map_frame = next_floor_start.to_map_frame((self.current_map.info.origin.position.x, self.current_map.info.origin.position.y),
                                                           self.current_map.info.resolution,
                                                           (self.current_map.info.height, self.current_map.info.width))

        response.next_floor_start = Pose()
        response.next_floor_start.position.x = start_in_map_frame[0]
        response.next_floor_start.position.y = start_in_map_frame[1]
        q = tf_transformations.quaternion_about_axis(start_in_map_frame[2], (0, 0, 1))
        response.next_floor_start.orientation.x = q[0]
        response.next_floor_start.orientation.y = q[1]
        response.next_floor_start.orientation.z = q[2]
        response.next_floor_start.orientation.w = q[3]

        self.get_logger().info("Start pos on next floor: " + str(next_floor) + " " + str(start_in_map_frame))
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
        self.get_logger().info("Moved to new position")

        # Load new map
        req = LoadMap.Request()
        req.map_url = self.map_paths[map_name]
        res = self.load_map_client.call(req)
        if not res:
            self.get_logger().error("Could not load map")
            return response

        self.current_map = res.map
        self.current_floor_name = map_name
        self.shg_planner.update_floor((self.current_map.info.origin.position.x, self.current_map.info.origin.position.y),
                                      self.current_map.info.resolution,
                                      (self.current_map.info.height, self.current_map.info.width),
                                      self.current_floor_name)
        self.get_logger().info("Map loaded")

        # Set initial pose
        # TODO: Adapt to new pose on new floor
        req = PoseWithCovarianceStamped()
        req.header.frame_id = "map"
        req.pose.pose.position.x = t.transform.translation.x
        req.pose.pose.position.y = t.transform.translation.y
        req.pose.pose.position.z = t.transform.translation.z
        req.pose.pose.orientation = t.transform.rotation
        self.initial_pose_publisher.publish(req)
        self.get_logger().info("Initial pose set")

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
        room_hierarchy = convert_map_pos_to_hierarchy(self.current_map, msg.point.x, msg.point.y,
                                                      self.shg_planner.graph._get_child(self.current_floor_name).watershed)
        hierarchy = [self.current_floor_name] + room_hierarchy

        self.get_logger().info("Graph hierarchy: " + str(hierarchy))

    def plan_path_callback(self, request: GetPlan.Request, response: GetPlan.Response) -> GetPlan.Response:
        start_time = time()
        start_pose = (request.start.pose.position.x, request.start.pose.position.y)
        goal_pose = (request.goal.pose.position.x, request.goal.pose.position.y)
        self.get_logger().info("Plan from (" + str(round(start_pose[0], 2)) + ", " + str(round(start_pose[1], 2)) +
                               ") to (" + str(round(goal_pose[0], 2)) + ", " + str(round(goal_pose[1], 2)) + ") floor: " + str(request.goal.header.frame_id))

        # TODO: Use correct floors for hierarchical planning
        start_floor = self.current_floor_name
        goal_floor = self.current_floor_name if request.goal.header.frame_id == "map" else request.goal.header.frame_id

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
