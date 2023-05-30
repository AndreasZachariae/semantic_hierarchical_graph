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
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.srv import LoadMap, ClearEntireCostmap
from geometry_msgs.msg import PoseStamped, Pose, PointStamped, PoseWithCovarianceStamped
from shg_interfaces.srv import GetElevatorData, ChangeMap
from std_msgs.msg import Empty

# For simulation teleport
from gazebo_msgs.srv import SetEntityState
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from semantic_hierarchical_graph.planners.shg_planner import SHGPlanner
from semantic_hierarchical_graph.types.exceptions import SHGPlannerError
from semantic_hierarchical_graph.types.position import Position, convert_map_pos_to_hierarchy
from semantic_hierarchical_graph.utils import path_to_list


class GraphNode(Node):

    def __init__(self):
        super().__init__('graph_node')  # type: ignore

        # Load parameters
        self.interpolation_resolution = self.declare_parameter(
            "interpolation_resolution", 1).get_parameter_value().double_value
        self.force_build_new_graph = self.declare_parameter(
            "force_build_new_graph", False).get_parameter_value().bool_value
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

        # Create graph
        self.get_logger().info("Graph name: " + str(self.graph_name) + ", initial floor: " + str(self.initial_map))
        self.shg_planner = SHGPlanner(graph_path, "graph.pickle", self.force_build_new_graph)

        # Create services and subscribers
        self.plan_srv = self.create_service(GetPlan, 'shg/plan_path', self.plan_path_callback)
        self.map_client = self.create_client(GetMap, 'map_server/map')
        self.point_sub = self.create_subscription(PointStamped, 'clicked_point', self.clicked_point_callback, 10)
        self.get_elevator_data_srv = self.create_service(
            GetElevatorData, 'shg/get_elevator_data', self.get_elevator_data_callback)
        self.change_map_srv = self.create_service(ChangeMap, 'shg/change_map', self.change_map_callback)

        # Create clients from within callbacks
        self.mutex_cb_group = MutuallyExclusiveCallbackGroup()
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10, callback_group=self.mutex_cb_group)
        self.clear_local_costmap_client = self.create_client(
            ClearEntireCostmap, "/local_costmap/clear_entirely_local_costmap", callback_group=self.mutex_cb_group)
        self.clear_global_costmap_client = self.create_client(
            ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap", callback_group=self.mutex_cb_group)
        self.load_map_client = self.create_client(
            LoadMap, "/map_server/load_map", callback_group=self.mutex_cb_group)

        # Only for tablet visualization
        self.current_pose_publisher = self.create_publisher(
            PoseStamped, "/shg/current_position", 10, callback_group=self.mutex_cb_group)

        # Initialize graph
        self.current_map = self.get_initial_map()
        self.current_floor_name = self.initial_map

        self.shg_planner.update_floor((self.current_map.info.origin.position.x, self.current_map.info.origin.position.y),
                                      self.current_map.info.resolution,
                                      (self.current_map.info.height, self.current_map.info.width),
                                      self.current_floor_name)

        # Services only for gazebo simulation teleport
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.gazebo_teleport_srv = self.create_service(ChangeMap, 'shg/gazebo_teleport', self.gazebo_teleport_callback)
        self.set_entity_state_client = self.create_client(
            SetEntityState, "/gazebo/set_entity_state", callback_group=self.mutex_cb_group)

        # Wait for external services to be ready
        while not (self.set_entity_state_client.wait_for_service(timeout_sec=1.0) and
                   self.load_map_client.wait_for_service(timeout_sec=1.0) and
                   self.clear_local_costmap_client.wait_for_service(timeout_sec=1.0) and
                   self.clear_global_costmap_client.wait_for_service(timeout_sec=1.0)):
            self.get_logger().info('Necessary services not available, waiting again...')

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
        if next_index >= len(floor_list):
            self.get_logger().info("No more floors to visit")
            return response
        next_floor = floor_list[next_index]
        # next_floor = self.current_floor_name
        response.next_floor = next_floor
        print("Next floor: " + str(next_floor))

        # Get marker id
        for map_config in self.graph_config["maps"]:
            if next_floor in map_config["hierarchy"]:
                floor_marker_id = map_config["marker_id"]
                break
        else:
            self.get_logger().error("Floor " + str(next_floor) + " not found in graph config")
            return response
        response.floor_marker_id = floor_marker_id
        print("Marker id: " + str(floor_marker_id))

        # Get elevator direction
        current_pos_z = self.shg_planner.graph.get_child_by_hierarchy([self.current_floor_name]).pos_abs.z
        next_pos_z = self.shg_planner.graph.get_child_by_hierarchy([next_floor]).pos_abs.z
        call_button_marker_id = "call_button_marker_id_up" if next_pos_z > current_pos_z else "call_button_marker_id_down"

        # Get elevator parameters
        current_path_list = self.shg_planner.get_path_on_floor([next_floor], "node", None)
        # import semantic_hierarchical_graph.utils as utils
        # print("current_path_list", utils.map_names_to_nodes(current_path_list))
        print(current_path_list[-1].data_dict)
        if current_path_list[-1].data_dict:
            # call_button_angle = current_path_list[-1].data_dict["call_button_angle"]
            response.call_button_marker_id = current_path_list[-1].data_dict[call_button_marker_id]
            response.waiting_start.position.x = current_path_list[-1].data_dict["waiting_pose_start_position"][0]
            response.waiting_start.position.y = current_path_list[-1].data_dict["waiting_pose_start_position"][1]
            response.waiting_start.position.z = current_path_list[-1].data_dict["waiting_pose_start_position"][2]
            response.waiting_start.orientation.x = current_path_list[-1].data_dict["waiting_pose_start_orientation"][0]
            response.waiting_start.orientation.y = current_path_list[-1].data_dict["waiting_pose_start_orientation"][1]
            response.waiting_start.orientation.z = current_path_list[-1].data_dict["waiting_pose_start_orientation"][2]
            response.waiting_start.orientation.w = current_path_list[-1].data_dict["waiting_pose_start_orientation"][3]
            response.waiting_goal.position.x = current_path_list[-1].data_dict["waiting_pose_goal_position"][0]
            response.waiting_goal.position.y = current_path_list[-1].data_dict["waiting_pose_goal_position"][1]
            response.waiting_goal.position.z = current_path_list[-1].data_dict["waiting_pose_goal_position"][2]
            response.waiting_goal.orientation.x = current_path_list[-1].data_dict["waiting_pose_goal_orientation"][0]
            response.waiting_goal.orientation.y = current_path_list[-1].data_dict["waiting_pose_goal_orientation"][1]
            response.waiting_goal.orientation.z = current_path_list[-1].data_dict["waiting_pose_goal_orientation"][2]
            response.waiting_goal.orientation.w = current_path_list[-1].data_dict["waiting_pose_goal_orientation"][3]
            response.panel_start.x = current_path_list[-1].data_dict["panel_point_start"][0]
            response.panel_start.y = current_path_list[-1].data_dict["panel_point_start"][1]
            response.panel_goal.z = current_path_list[-1].data_dict["panel_point_start"][2]
            response.panel_goal.x = current_path_list[-1].data_dict["panel_point_goal"][0]
            response.panel_goal.y = current_path_list[-1].data_dict["panel_point_goal"][1]
            response.panel_goal.z = current_path_list[-1].data_dict["panel_point_goal"][2]

            print("call_button_marker_id", response.call_button_marker_id)
            print("waiting_start (", response.waiting_start.x, response.waiting_start.y, ")")
            print("waiting_goal (", response.waiting_goal.x, response.waiting_goal.y, ")")
            print("panel_start (", response.panel_start.x, response.panel_start.y, response.panel_start.z, ")")
            print("panel_goal (", response.panel_goal.x, response.panel_goal.y, response.panel_goal.z, ")")

        # Get next floor start pose
        # path_list = self.shg_planner.get_path_on_floor([next_floor], "position", None)
        # if len(path_list) == 0:
        #     self.get_logger().error("No path found on selected floor, call 'shg/plan_path' service first " + str(next_floor))
        #     return response

        # next_floor_start: Position = path_list[0]
        # start_in_map_frame = next_floor_start.to_map_frame((self.current_map.info.origin.position.x, self.current_map.info.origin.position.y),
        #                                                    self.current_map.info.resolution,
        #                                                    (self.current_map.info.height, self.current_map.info.width))

        # response.next_floor_start = Pose()
        # response.next_floor_start.position.x = start_in_map_frame[0]
        # response.next_floor_start.position.y = start_in_map_frame[1]
        # q = tf_transformations.quaternion_about_axis(start_in_map_frame[2], (0, 0, 1))
        # response.next_floor_start.orientation.x = q[0]
        # response.next_floor_start.orientation.y = q[1]
        # response.next_floor_start.orientation.z = q[2]
        # response.next_floor_start.orientation.w = q[3]

        # self.get_logger().info("Start pos on next floor: " + str(next_floor) + " " + str(start_in_map_frame))

        return response

    def gazebo_teleport_callback(self, request: ChangeMap.Request, response: ChangeMap.Response) -> ChangeMap.Response:
        # Get current position
        # try:
        #     t = self.tf_buffer.lookup_transform(
        #         "map",
        #         "base_footprint",
        #         rclpy.time.Time())
        # except TransformException as ex:
        #     self.get_logger().error(
        #         f'Could not transform {"map"} to {"base_footprint"}: {ex}')
        #     return response

        # Get direction
        current_pos_z = self.shg_planner.graph.get_child_by_hierarchy([self.current_floor_name]).pos_abs.z
        next_pos_z = self.shg_planner.graph.get_child_by_hierarchy([request.map_name]).pos_abs.z
        offset_z = (next_pos_z - current_pos_z) * self.floor_height

        # Teleport robot to new floor
        new_position = SetEntityState.Request()
        new_position.state.name = "turtlebot3_waffle"
        new_position.state.pose = request.initial_pose
        new_position.state.pose.position.z += offset_z

        res = self.set_entity_state_client.call(new_position)
        if not res.success:
            self.get_logger().error("Could not set new position")
            return response
        self.get_logger().info("Moved to new position: (" + str(new_position.state.pose.position.x) + ", " +
                               str(new_position.state.pose.position.y) + ", " + str(new_position.state.pose.position.z) + ")")

        return response

    def change_map_callback(self, request: ChangeMap.Request, response: ChangeMap.Response) -> ChangeMap.Response:
        if request.map_name not in self.map_paths:
            self.get_logger().error("Map name not available, choose from: " + str(self.map_paths.keys()))
            return response
        self.get_logger().info("Changing map to: " + str(request.map_name))

        # Load new map
        req = LoadMap.Request()
        req.map_url = self.map_paths[request.map_name]
        res = self.load_map_client.call(req)
        if not res:
            self.get_logger().error("Could not load map")
            return response

        sleep(2)

        self.current_map = res.map
        self.current_floor_name = request.map_name
        self.shg_planner.update_floor((self.current_map.info.origin.position.x, self.current_map.info.origin.position.y),
                                      self.current_map.info.resolution,
                                      (self.current_map.info.height, self.current_map.info.width),
                                      self.current_floor_name)
        self.get_logger().info("Map loaded")

        # Set initial pose
        req = PoseWithCovarianceStamped()
        req.header.frame_id = "map"
        req.pose.pose = request.initial_pose
        self.initial_pose_publisher.publish(req)
        self.get_logger().info("Initial pose set")

        # Clear costmaps
        # TODO: Unnecessary because it is done in behavior tree again
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
        start_floor = self.current_floor_name
        goal_floor = self.current_floor_name if request.goal.header.frame_id == "map" else request.goal.header.frame_id

        self.get_logger().info("Plan from (" + str(round(start_pose[0], 2)) + ", " + str(round(start_pose[1], 2)) +
                               ") to (" + str(round(goal_pose[0], 2)) + ", " + str(round(goal_pose[1], 2)) + ") floor: " + str(goal_floor))

        path_list, distance = self.shg_planner.plan_in_map_frame(start_pose, start_floor,
                                                                 goal_pose, goal_floor,
                                                                 self.interpolation_resolution)

        for x, y, rz in path_list:
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            if tuple([x, y, rz]) == path_list[-1] and rz is None:
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

        self.current_pose_publisher.publish(request.start)

        self.get_logger().info("Path length: " + str(round(distance, 2)) + ", nodes:" + str(len(response.plan.poses)))
        if planning_time > 1.0:
            self.get_logger().warn("Planning time: " + str(round(planning_time, 6)))
        else:
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
