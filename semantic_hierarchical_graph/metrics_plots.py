from matplotlib import pyplot as plt
import numpy as np
import json


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def plot_floor_metrics(metrics, path):
    names = ["SHG", "AStar", "PRM", "RRT"]
    planners = [metrics[name] for name in names]
    success_rate = [planner["success_rate"] for planner in planners]
    planning_time = [planner["planning_time"]["mean"] for planner in planners]
    path_length = [planner["path_length"]["mean"] for planner in planners]
    smoothness = [planner["smoothness"]["mean"] for planner in planners]
    obstacle_clearance = [planner["obstacle_clearance"]["mean"] for planner in planners]
    distance_std = [planner["obstacle_distance_std"]["mean"] for planner in planners]

    metric_names = ["Success rate", "Planning time", "Path length", "Smoothness",
                    "Obstacle clearance", "Distance std"]
    metrics = [success_rate, planning_time, path_length, smoothness,
               obstacle_clearance, distance_std]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:gray']
    units = ["%", "s", "px", "°/px", "px", "px"]
    y_scale_to_1 = [True, False, False, False, False, False]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

    for axis, metric, metric_name, color, unit, ylim in zip(np.ravel(axes), metrics,  # type: ignore
                                                            metric_names, colors, units, y_scale_to_1):
        axis.bar(names, metric, label=metric_name, color=color)
        axis.set_title(metric_name)
        # axis.set_xticklabels(names, rotation=45)
        if ylim:
            axis.set_ylim(bottom=0, top=1)
        axis.set_ylabel(f"{metric_name} [{unit}]", color=color)

    fig.tight_layout()
    # plt.show()
    plt.savefig(path)


def plot_room_metrics(metrics, path):
    names = ["ILIR", "AStar", "PRM", "RRT"]
    planners = [metrics[name] for name in names]
    success_rate = [planner["success_rate"] for planner in planners]
    planning_time = [planner["planning_time"]["mean"] for planner in planners]
    path_length = [planner["path_length"]["mean"] for planner in planners]
    smoothness = [planner["smoothness"]["mean"] for planner in planners]
    obstacle_clearance = [planner["obstacle_clearance"]["mean"] for planner in planners]
    disturbance = [planner["disturbance"] for planner in planners]
    distance_std = [planner["obstacle_distance_std"]["mean"] for planner in planners]
    distance_centroid = [planner["centroid_distance"]["mean"] for planner in planners]

    metric_names = ["Success rate", "Planning time", "Path length", "Smoothness",
                    "Obstacle clearance", "Distance std", "Distance to centroid", "Disturbance"]
    metrics = [success_rate, planning_time, path_length, smoothness,
               obstacle_clearance, distance_std, distance_centroid, disturbance]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:gray', 'tab:brown', 'tab:olive']
    units = ["%", "s", "px", "°/px", "px", "px", "px", "%"]
    y_scale_to_1 = [True, False, False, False, False, False, False, True]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))

    for axis, metric, metric_name, color, unit, ylim in zip(np.ravel(axes), metrics,  # type: ignore
                                                            metric_names, colors, units, y_scale_to_1):
        axis.bar(names, metric, label=metric_name, color=color)
        axis.set_title(metric_name)
        # axis.set_xticklabels(names, rotation=45)
        if ylim:
            axis.set_ylim(bottom=0, top=1)
        axis.set_ylabel(f"{metric_name} [{unit}]", color=color)

    fig.tight_layout()
    # plt.show()
    plt.savefig(path)


if __name__ == "__main__":
    metrics = load_json("data/floor_ryu_metrics_ALL.json")
    plot_floor_metrics(metrics, "data/floor_ryu_metrics.png")
