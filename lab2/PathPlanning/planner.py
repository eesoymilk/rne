import abc
from numpy.typing import NDArray


class Planner:

    def __init__(self, m: NDArray):
        self.map = m

    @abc.abstractmethod
    def planning(self, start, goal):
        return NotImplementedError
