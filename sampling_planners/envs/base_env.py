from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class BaseEnv(ABC):
    @abstractmethod
    def is_state_valid(self, state: np.ndarray) -> bool:
        pass

    @abstractmethod
    def distance(s1: np.ndarray, s2: np.ndarray) -> float:
        pass

    @abstractmethod
    def draw_random_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def visualize(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        tree_nodes: Sequence[Sequence[np.ndarray]],
        tree_edges: Sequence[Sequence[tuple[int, int]]],
        highlighted_path: Sequence[Sequence[tuple[int, int]] | None] | None = None,
        filename: str = "",
    ):
        """Visualize the sampling trees and found trajectory, if any.

        This method creates a visualization of the environment, resulted tree in this trajectory, and
        the found path, if any.

        Args:
            start (np.ndarray): starting state of the trajectory.

            goal (np.ndarray): goal state of the trajectory.

            tree_nodes (Sequence[ Sequence[np.ndarray]]): nodes of the tree.
            Nodes sequence is a Sequence[Sequence[np.ndarray]]. Nodes are only added to the tree,
            hence each entry in the sequence represents only new nodes added to the tree during one iteration.
            The overall list of nodes in a tree is prefix sum of all lists in Sequence.

            tree_edges (Sequence[tuple[int, int]]): edges of the trees.
            Each edge is a tuple of (node1, node2), and each iteration contains Sequence[tuple[int, int]].
            Unlike in the trajectory, the edges could dissappear from the tree between iterations.

            highlighted_path (Sequence[Sequence[int] | None] | None): hightlighted path in the tree.
            If highlighted path is present, it is a list of edges in the tree. Otherwise, it is None.
            Defaults to None, meaning no path highlighted.

            filename (str, optional): name of the file where to save the result of visualization.
            Defaults to "", meaning no visualization.
        """
