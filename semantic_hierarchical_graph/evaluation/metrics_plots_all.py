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
    success_rate = [planner["success_rate"]*100 for planner in planners]
    planning_time = [planner["planning_time"]["mean"] for planner in planners]
    path_length = [planner["path_length"]["mean"] for planner in planners]
    smoothness = [planner["smoothness"]["mean"] for planner in planners]
    obstacle_clearance = [planner["obstacle_clearance"]["mean"] for planner in planners]
    disturbance = [round(planner["disturbance"]*100) for planner in planners]
    distance_std = [planner["obstacle_distance_std"]["mean"] for planner in planners]
    distance_centroid = [planner["centroid_distance"]["mean"] for planner in planners]

    metric_names = ["Planning time", "Path length", "Disturbance",
                    "Obstacle clearance", "Obstacle clearance std", "Distance to centroid"]
    metrics = [planning_time, path_length,disturbance,
               obstacle_clearance, distance_std, distance_centroid ]
    colors = [ 'tab:orange', 'tab:green', 'darkcyan', 'tab:gray', 'tab:brown', 'tab:olive']
    units = [ "s", "px", "%", "px", "px", "px"]
    y_scale_to_1 = [False, False, True, False, False, False]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

    for axis, metric, metric_name, color, unit, ylim in zip(np.ravel(axes), metrics,  # type: ignore
                                                            metric_names, colors, units, y_scale_to_1):
        axis.bar(names, metric, label=metric_name, color=color)
        axis.set_title(metric_name)
        # axis.set_xticklabels(names, rotation=45)
        if ylim:
            axis.set_ylim(bottom=0, top=100)
        axis.set_ylabel(f"{metric_name} [{unit}]", color=color)
        axis.bar_label(axis.containers[0], label_type='edge')

    fig.tight_layout()
    # plt.show()
    plt.savefig(path)


if __name__ == "__main__":
    metrics = load_json("data/ryu_room2_metrics.json")
    plot_room_metrics(metrics, "data/ryu_room2_metrics.png")
