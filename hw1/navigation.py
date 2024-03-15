import argparse
import numpy as np
import cv2
from typing import Optional
from numpy.typing import NDArray
from cv2.typing import MatLike
from Simulation.utils import ControlState
from Simulation.simulator_map import SimulatorMap
from PathPlanning.cubic_spline import cubic_spline_2d, pos_int


class Navigator:
    simulator_map_params = {
        "basic": {
            "l": 9,
            "wu": 7,
            "wv": 3,
            "car_w": 16,
            "car_f": 13,
            "car_r": 7,
        },
        "diff_drive": {
            "l": 9,
            "wu": 7,
            "wv": 3,
            "car_w": 16,
            "car_f": 13,
            "car_r": 7,
        },
        "bicycle": {
            "l": 20,
            "d": 5,
            "wu": 5,
            "wv": 2,
            "car_w": 14,
            "car_f": 25,
            "car_r": 5,
        },
    }

    def __init__(self, args: argparse.Namespace, m: MatLike):
        self.simulator_name: str = args.simulator
        self.controller_name: str = args.controller
        self.planner_name: str = args.planner
        self._map = m
        self._map_cspace = 1 - cv2.dilate(1 - m, np.ones((40, 40)))
        self._set_controller_path = False
        self._nav_pos: Optional[tuple[int, int]] = None
        self._path: Optional[NDArray] = None
        self._pose: Optional[tuple[float, float, float]] = None
        self._way_points: Optional[NDArray] = None

        controller_params = {}
        if self.simulator_name == "basic":
            from Simulation.simulator_basic import (
                SimulatorBasic as Simulator,
            )

            if self.controller_name == "pid":
                from PathTracking.controller_pid_basic import (
                    ControllerPIDBasic as Controller,
                )
            elif self.controller_name == "pure_pursuit":
                from PathTracking.controller_pure_pursuit_basic import (
                    ControllerPurePursuitBasic as Controller,
                )

                controller_params["Lfc"] = 1
            elif self.controller_name == "lqr":
                from PathTracking.controller_lqr_basic import (
                    ControllerLQRBasic as Controller,
                )
            else:
                raise NameError("Unknown controller!!")
        elif self.simulator_name == "diff_drive":
            from Simulation.simulator_differential_drive import (
                SimulatorDifferentialDrive as Simulator,
            )

            if self.controller_name == "pid":
                from PathTracking.controller_pid_basic import (
                    ControllerPIDBasic as Controller,
                )
            elif self.controller_name == "pure_pursuit":
                from PathTracking.controller_pure_pursuit_basic import (
                    ControllerPurePursuitBasic as Controller,
                )

                controller_params["Lfc"] = 1
            elif self.controller_name == "lqr":
                from PathTracking.controller_lqr_basic import (
                    ControllerLQRBasic as Controller,
                )
            else:
                raise NameError("Unknown controller!!")

        elif self.simulator_name == "bicycle":
            from Simulation.simulator_bicycle import (
                SimulatorBicycle as Simulator,
            )

            if self.controller_name == "pid":
                from PathTracking.controller_pid_bicycle import (
                    ControllerPIDBicycle as Controller,
                )
            elif self.controller_name == "pure_pursuit":
                from PathTracking.controller_pure_pursuit_bicycle import (
                    ControllerPurePursuitBicycle as Controller,
                )

                controller_params["Lfc"] = 1
            elif self.controller_name == "stanley":
                from PathTracking.controller_stanley_bicycle import (
                    ControllerStanleyBicycle as Controller,
                )
            elif self.controller_name == "lqr":
                from PathTracking.controller_lqr_bicycle import (
                    ControllerLQRBicycle as Controller,
                )
            else:
                raise NameError("Unknown controller!!")
        else:
            raise NameError("Unknown simulator!!")

        if self.planner_name == "a_star":
            from PathPlanning.planner_a_star import PlannerAStar as Planner
        elif self.planner_name == "rrt":
            from PathPlanning.planner_rrt import PlannerRRT as Planner
        elif self.planner_name == "rrt_star":
            from PathPlanning.planner_rrt_star import (
                PlannerRRTStar as Planner,
            )
        else:
            raise NameError("Unknown planner!!")

        self.simulator = SimulatorMap(
            Simulator,
            m=m,
            **self.simulator_map_params[args.simulator],
        )
        self.controller = Controller(**controller_params)
        self.planner = Planner(self._map_cspace)

    def run(self, start_pose=(100, 200, 0)):
        def mouse_click(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONUP:
                return

            nav_pos_new = (x, self._map.shape[0] - y)
            if self._map_cspace[nav_pos_new[1], nav_pos_new[0]] <= 0.5:
                return

            self._way_points = self.planner.planning(
                (pose[0], pose[1]), nav_pos_new, 20
            )
            if len(self._way_points) == 0:
                return

            self._nav_pos = nav_pos_new
            self._path = np.array(cubic_spline_2d(self._way_points, interval=1))

            self._set_controller_path = True

        # Initialize
        window_name = "Known Map Navigation Demo"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_click)
        self.simulator.init_pose(start_pose)
        command = ControlState(self.simulator_name, None, None)
        pose = start_pose
        collision_count = 0
        # Main Loop
        while True:
            # Update State
            self.simulator.step(command)
            pose = (
                self.simulator.state.x,
                self.simulator.state.y,
                self.simulator.state.yaw,
            )
            # print("\r", self.simulator, "| Goal:", self._nav_pos, end="\t")

            if self._set_controller_path:
                self.controller.set_path(self._path)
                self._set_controller_path = False

            if self._path is not None and collision_count == 0:
                end_dist = np.hypot(
                    self._path[-1, 0] - self.simulator.state.x,
                    self._path[-1, 1] - self.simulator.state.y,
                )
                # TODO: Planning and Controlling
                if self.simulator_name == "basic":
                    if end_dist > 10:
                        next_v = 2
                    else:
                        next_v = 0
                    # Lateral
                    info = {
                        "x": self.simulator.state.x,
                        "y": self.simulator.state.y,
                        "yaw": self.simulator.state.yaw,
                        "v": self.simulator.state.v,
                        "dt": self.simulator.dt,
                    }
                    next_w, target = self.controller.feedback(info)
                    command = ControlState("basic", next_v, next_w)
                elif self.simulator_name == "diff_drive":
                    # Longitude
                    if end_dist > 10:
                        next_v = 5
                    else:
                        next_v = 0
                    # Lateral
                    info = {
                        "x": self.simulator.state.x,
                        "y": self.simulator.state.y,
                        "yaw": self.simulator.state.yaw,
                        "v": self.simulator.state.v,
                        "dt": self.simulator.dt,
                    }
                    next_w, target = self.controller.feedback(info)
                    # TODO: v,w to motor control
                    r, l = self.simulator.wu / 2, self.simulator.l / 2
                    next_lw = np.rad2deg(
                        next_v / r - np.deg2rad(next_w) * l / r
                    )
                    next_rw = np.rad2deg(
                        next_v / r + np.deg2rad(next_w) * l / r
                    )
                    command = ControlState("diff_drive", next_lw, next_rw)
                elif self.simulator_name == "bicycle":
                    # Longitude (P Control)
                    if end_dist > 40:
                        target_v = 5
                    else:
                        target_v = 0
                    next_a = (target_v - self.simulator.state.v) * 0.5
                    # Lateral
                    info = {
                        "x": self.simulator.state.x,
                        "y": self.simulator.state.y,
                        "yaw": self.simulator.state.yaw,
                        "v": self.simulator.state.v,
                        "delta": self.simulator.cstate.delta,
                        "l": self.simulator.l,
                        "dt": self.simulator.dt,
                    }
                    next_delta, target = self.controller.feedback(info)
                    command = ControlState("bicycle", next_a, next_delta)
                else:
                    exit()
            else:
                command = None

            _, info = self.simulator.step(command)
            # Collision Handling
            if info["collision"]:
                collision_count = 1
            if collision_count > 0:
                # TODO: Collision Handling
                pass

            # Render Path
            img = self.simulator.render()
            if self._nav_pos is not None and self._way_points is not None:
                img = self._render_path(img)

            img = cv2.flip(img, 0)
            cv2.imshow(window_name, img)
            k = cv2.waitKey(1)
            if k == ord('r'):
                self.simulator.init_state(start_pose)
            if k == 27:
                print()
                break

    def _render_path(self, img: MatLike):
        cv2.circle(img, self._nav_pos, 5, (0.5, 0.5, 1.0), 3)
        for i in range(len(self._way_points)):  # Draw Way Points
            cv2.circle(img, pos_int(self._way_points[i]), 3, (1.0, 0.4, 0.4), 1)
        for i in range(len(self._path) - 1):  # Draw Interpolating Curve
            cv2.line(
                img,
                pos_int(self._path[i]),
                pos_int(self._path[i + 1]),
                (1.0, 0.4, 0.4),
                1,
            )
        return img


def main(args: argparse.Namespace):
    # Read Map
    img = cv2.flip(cv2.imread(args.map), 0)
    img[img > 128] = 255
    img[img <= 128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.0

    navigator = Navigator(args, m)
    navigator.run()


if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--simulator",
        type=str,
        default="diff_drive",
        help="diff_drive/bicycle",
    )
    parser.add_argument(
        "-c",
        "--controller",
        type=str,
        default="pure_pursuit",
        help="pid/pure_pursuit/stanley/lqr",
    )
    parser.add_argument(
        "-p",
        "--planner",
        type=str,
        default="a_star",
        help="a_star/rrt/rrt_star",
    )
    parser.add_argument(
        "-m", "--map", type=str, default="Maps/map1.png", help="image file name"
    )
    args = parser.parse_args()

    main(args)
