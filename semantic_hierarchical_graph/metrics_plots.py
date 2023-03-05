from matplotlib import pyplot as plt
import numpy as np
import json

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_metrics(metrics):
    names = ["ILIR", "AStar", "PRM", "RRT"]
    planners = [metrics[name] for name in names]
    success_rate = [planner["success_rate"] for planner in planners]
    planning_time = [planner["planning_time"]["mean"] for planner in planners]
    path_length = [planner["path_length"]["mean"] for planner in planners]
    smoothness = [planner["smoothness"]["mean"] for planner in planners]
    obstacle_clearance = [planner["obstacle_clearance"]["mean"] for planner in planners]
    disturbance = [planner["disturbance"] for planner in planners]

    metric_names = ["Success rate", "Planning time", "Path length", "Smoothness", "Obstacle clearance", "Disturbance"]
    metrics = [success_rate, planning_time, path_length, smoothness, obstacle_clearance, disturbance]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:olive']
    units = ["%", "s", "px", "°/px", "px", "%"]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    for axis, metric ,metric_name, color, unit in zip(np.ravel(axes), metrics, metric_names, colors, units): # type: ignore
        axis.bar(names, metric, label=metric_name, color=color)
        axis.set_title(metric_name)
        # axis.set_xticklabels(names, rotation=45)
        axis.set_ylabel(f"{metric_name} [{unit}]", color=color)
        
    fig.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    metrics = load_json("data/ryu_room2_metrics_no_smooth.json")
    plot_metrics(metrics)