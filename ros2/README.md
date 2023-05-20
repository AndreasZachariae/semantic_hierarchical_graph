# ROS2 Graph node and custom SHGPlanner

## How to start

```bash
source start_docker.sh
ros2 launch shg tb3_simulation_launch.py
ros2 launch shg aws_hospital_simulation.launch.py
ros2 launch shg graph.launch.py
```

## Test ROS2 services:

```bash
# SHG gazebo_teleport
ros2 service call /shg/gazebo_teleport shg_interfaces/srv/ChangeMap "{map_name: aws2, initial_pose: {position: {x: 3, y: 2}}}"

# SHG change_map
ros2 service call /shg/change_map shg_interfaces/srv/ChangeMap "{map_name: aws2, initial_pose: {position: {x: 3, y: 2}}}"

# Gazebo get_entity_state
ros2 service call /gazebo/get_entity_state gazebo_msgs/srv/GetEntityState "{name: turtlebot3_waffle}"

# Gazebo set_entity_state
ros2 service call /gazebo/set_entity_state gazebo_msgs/srv/SetEntityState "{state: {name: turtlebot3_waffle, pose: {position: {x: 3, y: 2}}}}"
```

## Change planners of Nav2

in `ros2/config/nav2_params.yaml`  
This is the default config:
```yaml
planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
```
Change planner plugin to 
```yaml
plugin: "shg/SHGPlanner"
```