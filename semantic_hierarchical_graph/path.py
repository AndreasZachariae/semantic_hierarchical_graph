class SHPath():
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.all_paths = []
        self.path_checked = False

    def all_paths_checked(self):
        val = self.path_checked
        self.path_checked = True
        return val

    def get_shortest_path(self):
        return self.all_paths[0]
