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
        self.parent: dict[tuple[int, int], int | None] = dict()
        # Distance from start to node
        self.h: dict[tuple[int, int], int] = dict()
        # Distance from node to goal
        self.g: dict[tuple[int, int], int] = dict()
        self.goal_node = None

    def planning(self, start=(100, 200), goal=(375, 520), inter=None, img=None):
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

        cnt = 0
        while True:
            current = self.queue.pop(0)
            print(f"processing {current}...")
            if current == goal:
                self.goal_node = current
                break

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                print(f"neighbor: {neighbor}")
                if not self._is_valid_node(current, neighbor):
                    continue
                print(f"valid neighbor: {neighbor}")

                tmp_g = self.g[current] + utils.distance(current, neighbor)
                if neighbor in self.g and tmp_g > self.g[neighbor]:
                    continue

                self.parent[neighbor] = current
                self.g[neighbor] = tmp_g
                self.h[neighbor] = utils.distance(neighbor, goal)
                self.queue.append(neighbor)

            self.queue.sort(key=lambda x: self.g[x] + self.h[x])
            # print("queue:")
            # pprint(self.queue)
            # print("parent:")
            # pprint(self.parent)
            # print("g:")
            # pprint(self.g)
            # print("h:")
            # pprint(self.h)

            # cnt += 1
            # if cnt > 1:
            #     break
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

    def _is_valid_node(
        self,
        current_node: tuple[int, int],
        next_node: tuple[int, int],
    ):
        line = utils.Bresenham(
            current_node[0], next_node[0], current_node[1], next_node[1]
        )
        x_coords, y_coords = zip(*line)
        x_array = np.array(x_coords)
        y_array = np.array(y_coords)
        if next_node == (120, 220):
            print(f"{self.parent[current_node] != next_node=}")
            print(f"{0 <= next_node[0] < self.map.shape[0]=}")
            print(f"{0 <= next_node[1] < self.map.shape[1]=}")
            print(f"{np.all(self.map[x_array, y_array] == 0)=}")
            print(f"{line=}")
            print(f"{self.map[x_array, y_array]=}")
        return (
            self.parent[current_node] != next_node
            and 0 <= next_node[0] < self.map.shape[0]
            and 0 <= next_node[1] < self.map.shape[1]
            and np.all(self.map[x_array, y_array] == 0)
        )
