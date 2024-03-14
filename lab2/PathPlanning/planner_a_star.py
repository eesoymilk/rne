import cv2
import sys
import numpy as np
from pprint import pprint

sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner


class PlannerAStar(Planner):
    def __init__(self, m, inter=10):
        super().__init__(m)
        self.inter = inter
        self.initialize()

    def initialize(self):
        self.queue: list[tuple[int, int]] = []
        self.parent: dict[tuple[int, int], tuple[int, int] | None] = dict()
        # Distance from start to node
        self.h: dict[tuple[int, int], int] = dict()
        # Distance from node to goal
        self.g: dict[tuple[int, int], int] = dict()
        self.goal_node = None

    def planning(
        self, start=(100, 200), goal=(375, 520), inter=None, img=None
    ):
        if inter is None:
            inter = self.inter

        directions = [
            (-inter, 0),
            (inter, 0),
            (0, -inter),
            (0, inter),
            (-inter, -inter),
            (inter, inter),
            (-inter, inter),
            (inter, -inter),
        ]

        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        self.initialize()
        self.queue.append(start)
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal)

        while self.queue:
            current = self.queue.pop(0)
            if current == goal:
                self.goal_node = current
                break

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self._is_valid_node(current, neighbor):
                    continue

                tmp_g = self.g[current] + utils.distance(current, neighbor)
                if neighbor in self.g and tmp_g >= self.g[neighbor]:
                    continue

                self.parent[neighbor] = current
                self.g[neighbor] = tmp_g

                if neighbor not in self.h:
                    self.h[neighbor] = utils.distance(neighbor, goal)

                if neighbor not in self.queue:
                    self.queue.append(neighbor)

            self.queue.sort(key=lambda x: self.g[x] + self.h[x])

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

    def _is_valid_node(
        self,
        current_node: tuple[int, int],
        next_node: tuple[int, int],
    ):
        x0, y0 = current_node
        x1, y1 = next_node
        height, width = self.map.shape
        if (
            current_node == next_node
            or next_node in self.parent
            or x0 < 0
            or y0 < 0
            or x1 < 0
            or y1 < 0
            or x0 >= width
            or y0 >= height
            or x1 >= width
            or y1 >= height
        ):
            return False

        line = utils.Bresenham(
            current_node[0], next_node[0], current_node[1], next_node[1]
        )
        x_coords, y_coords = zip(*line)
        if np.any(self.map[np.array(y_coords), np.array(x_coords)] == 0):
            return False

        return True
