import sys
import numpy as np

sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller


class ControllerStanleyBicycle(Controller):
    def __init__(self, kp=0.5):
        self.path = None
        self.kp = kp

    # State: [x, y, yaw, delta, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None

        # Extract State
        x, y, yaw, delta, v, l = (
            info["x"],
            info["y"],
            info["yaw"],
            info["delta"],
            info["v"],
            info["l"],
        )

        # Search Front Wheel Target
        front_x = x + l * np.cos(np.deg2rad(yaw))
        front_y = y + l * np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta))
        min_idx, min_dist = utils.search_nearest(self.path, (front_x, front_y))
        target = self.path[min_idx]

        # TODO: Stanley Control for Bicycle Kinematic Model
        if min_idx + 1 < len(self.path):
            theta_p = np.arctan2(
                self.path[min_idx + 1, 1] - target[1],
                self.path[min_idx + 1, 0] - target[0],
            )
        else:
            theta_p = np.arctan2(
                target[1] - self.path[min_idx - 1, 1],
                target[0] - self.path[min_idx - 1, 0],
            )
        theta_e = theta_p - np.deg2rad(yaw)
        vf_safe = max(vf, 0.001)
        err_dist = np.dot(
            [x - target[0], y - target[1]],
            [np.cos(theta_p + np.pi / 2), np.sin(theta_p + np.pi / 2)],
        )
        next_delta = np.rad2deg(
            np.arctan2(-self.kp * err_dist, vf_safe) + theta_e
        )
        return utils.angle_norm(next_delta), target
