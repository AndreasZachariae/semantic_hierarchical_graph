class ILIRPlanner():
    def __init__(self, graph, env):
        self.graph = graph
        self.env = env

    def plan(self, start, goal):
        if start not in graph.nodes:
            connect_to_graph(start)
        if goal not in graph.nodes:
            connect_to_graph(goal)

    def _connect_to_graph(self, node):
        pass
