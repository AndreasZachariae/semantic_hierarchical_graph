# ROS2 Graph node and custom SHGPlanner

## How to start

```bash
source start_docker.sh
ros2 launch shg tb3_simulation_launch.py
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