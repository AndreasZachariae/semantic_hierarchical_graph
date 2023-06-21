#!/bin/sh
uid=$(eval "id -u")
gid=$(eval "id -g")
docker build \
    --build-arg UID="$uid" \
    --build-arg GID="$gid" \
    -t shg_neobotix/ros:foxy .
