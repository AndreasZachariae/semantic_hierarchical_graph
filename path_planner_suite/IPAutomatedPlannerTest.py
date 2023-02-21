import matplotlib.pyplot as plt

from path_planner_suite.IPPerfMonitor import IPPerfMonitor
import path_planner_suite.IPAStar as IPAStar
import path_planner_suite.IPVISAStar as IPVISAStar
import path_planner_suite.IPTestSuite as IPTestSuite
from path_planner_suite.IPBenchmark import ResultCollection

if __name__ == "__main__":
    # Create dict of planners
    plannerFactory = dict()
    astarConfig = dict()
    astarConfig["heuristic"] = 'euclidean'
    astarConfig["w"] = 0.5
    plannerFactory["astar"] = [IPAStar.AStar, astarConfig, IPVISAStar.aStarVisualize]

    # Plan every planner on every benchmark
    resultList = list()
    for key, producer in list(plannerFactory.items()):
        print(key, producer)
        for benchmark in IPTestSuite.benchList:  # [0:24]
            print("Planning: " + key + " - " + benchmark.name)
            planner = producer[0](benchmark.collisionChecker)
            IPPerfMonitor.clearData()

            resultList.append(ResultCollection(key,
                                               planner,
                                               benchmark,
                                               planner.planPath(benchmark.startList, benchmark.goalList, producer[1]),
                                               IPPerfMonitor.dataFrame()))
    # Visualize results
    for result in resultList:
        if result.solution is None:
            print("No path found for " + result.benchmark.name)
            continue
        fig_local = plt.figure(figsize=(10, 10))
        ax = fig_local.add_subplot(1, 1, 1)
        title = result.plannerFactoryName + " - " + result.benchmark.name
        title += "\n Assumed complexity level " + str(result.benchmark.level)
        ax.set_title(title, color='w')

        plannerFactory[result.plannerFactoryName][2](result.planner, result.solution, ax=ax, nodeSize=100)
        plt.show()
