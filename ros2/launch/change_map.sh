#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <map_name>"
    return 1
fi

map_name="$1"
map_url="/home/docker/ros2_ws/install/shg/../../src/semantic_hierarchical_graph/ros2/config/./../../data/graphs/simulation/./floor/./${map_name}.yaml"

ros2 service call /map_server/load_map2 nav2_msgs/srv/LoadMap "{map_url: $map_url}"
