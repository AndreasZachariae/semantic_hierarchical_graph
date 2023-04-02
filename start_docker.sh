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
    -e DISPLAY=$DISPLAY \
    --rm \
    -v ./semantic_hierarchical_graph:/home/docker/ros2_ws/src/semantic_hierarchical_graph/semantic_hierarchical_graph:rw \
    -v ./path_planner_suite:/home/docker/ros2_ws/src/semantic_hierarchical_graph/path_planner_suite:rw \
    -v ./ros2:/home/docker/ros2_ws/src/semantic_hierarchical_graph/ros2:rw \
    -v ./data/graphs:/home/docker/ros2_ws/src/semantic_hierarchical_graph/data/graphs:rw \
    shg/ros:foxy
