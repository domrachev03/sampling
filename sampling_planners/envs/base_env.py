from abc import ABC, abstractmethod

import numpy as np


class BaseEnv(ABC):
    @abstractmethod
    def is_state_valid(self, state: np.ndarray) -> bool:
        pass

    @abstractmethod
    def distance(s1: np.ndarray, s2: np.ndarray) -> float:
        pass

    @abstractmethod
    def visualize(self, traj: np.ndarray, tree: list[np.ndarray], filename: str = "") -> None:
        pass

    @abstractmethod
    def draw_random_state(self) -> np.ndarray:
        pass
