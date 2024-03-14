import cv2
import sys

sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner


class PlannerAStar(Planner):
    def __init__(self, m, inter=10):
        super().__init__(m)
        self.inter = inter
        self.initialize()

    def initialize(self):
        self.queue = []
        self.parent: dict[tuple[int, int], int | None] = dict()
        # Distance from start to node
        self.h: dict[tuple[int, int], int] = dict()
        # Distance from node to goal
        self.g: dict[tuple[int, int], int] = dict()
        self.goal_node = None

    def planning(self, start=(100, 200), goal=(375, 520), inter=None, img=None):
        if inter is None:
            inter = self.inter

        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        self.initialize()
        self.queue.append(start)
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal)

        while True:
            # TODO: A Star Algorithm
            break

        # Extract path
        path = []
        p = self.goal_node
        if p is None:
            return path
        while True:
            path.insert(0, p)
            if self.parent[p] is None:
                break
            p = self.parent[p]
        if path[-1] != goal:
            path.append(goal)
        return path
