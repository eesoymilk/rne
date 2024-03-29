import numpy as np
import sys

sys.path.append("..")
from Simulation.utils import State, ControlState
from Simulation.kinematic import KinematicModel


class KinematicModelBicycle(KinematicModel):
    def __init__(self, l=30, dt=0.1):  # distance between rear and front wheel
        # Distance from center to wheel
        self.l = l
        # Simulation delta time
        self.dt = dt

    def step(self, state: State, cstate: ControlState) -> State:
        # TODO: Bicycle Kinematic Model
        v = state.v + cstate.a * self.dt
        w = np.rad2deg(state.v / self.l * np.tan(np.deg2rad(cstate.delta)))
        x = state.x + v * np.cos(np.deg2rad(state.yaw)) * self.dt
        y = state.y + v * np.sin(np.deg2rad(state.yaw)) * self.dt
        yaw = (state.yaw + w * self.dt) % 360
        state_next = State(x, y, yaw, v, w)
        return state_next
