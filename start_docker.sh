#!/bin/sh
uid=$(eval "id -u")
gid=$(eval "id -g")
docker build \
    --build-arg UID="$uid" \
    --build-arg GID="$gid" \
    -t shg/ros:foxy . && \

echo "Run Container" && \
xhost + local:root && \
docker run \
    --name semantic_hierarchical_graph \
    -it \
    --privileged \
    --net host \
    --env-file .env \
    -v semantic_hierarchical_graph:/home/ros2_ws/src/semantic_hierarchical_graph/semantic_hierarchical_graph \
    -v path_planner_suite:/home/ros2_ws/src/semantic_hierarchical_graph/path_planner_suite \
    -v ros2:/home/ros2_ws/src/semantic_hierarchical_graph/ros2 \
    -e DISPLAY=$DISPLAY \
    --rm shg/ros:foxy
