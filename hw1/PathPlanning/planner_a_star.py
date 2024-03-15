import cv2
import sys
import numpy as np

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
        self.parent = {}
        self.h = {}  # Distance from start to node
        self.g = {}  # Distance from node to goal
        self.goal_node = None

    def planning(
        self,
        start=(100, 200),
        goal=(375, 520),
        inter: int | None = None,
        img=None,
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
                neighbor = self._steer(current, (dx, dy), goal)
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

    def _steer(
        self,
        current: tuple[int, int],
        direction: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[int, int]:
        goal_vect = np.array(goal) - np.array(current)
        goal_dist = np.linalg.norm(goal_vect)

        if goal_dist < self.inter:
            dir_norm = np.array(direction) / np.linalg.norm(direction)
            angle = np.arccos(
                np.dot(dir_norm, goal_vect)
                / (np.linalg.norm(dir_norm) * goal_dist)
            )
            if angle < np.pi / 4:
                # Move towards the goal.
                return goal

        # Return the new position as a tuple of integers.
        return current[0] + direction[0], current[1] + direction[1]

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
