import numpy as np

from sampling_planners.envs.base_env import BaseEnv
from sampling_planners.planners.base_planner import BaseTreePlanner


class PrimitiveTreePlanner(BaseTreePlanner):
    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        env: BaseEnv,
        sol_threshold: float = 1e-3,
        extend_step: float = 0.1,
        choose_nearest: bool = False,
    ):
        super().__init__(start, goal, env, sol_threshold, extend_step=extend_step)
        self.choose_nearest = choose_nearest

    def step(self) -> bool:
        super().step()
        # Step 2. Selecting a random node from the tree
        init_node = None
        if self.choose_nearest:
            # Step 2.1. Choose the nearest node in the tree
            init_node = self.nearest_node(self.goal)
        else:
            init_node = np.random.randint(0, len(self.tree_nodes))
        random_state = self.tree_nodes[init_node]

        # Step 3. Select a new random environment collision-free state
        new_state = self.sample_state()

        # Step 4. Extend the tree towards the new state
        extended_state = self.extend(random_state, new_state)

        # Step 5. Update the tree with the new state and edges
        if len(extended_state) == 0:
            return False
        match len(extended_state):
            case 0:
                # No new nodes added
                return False

            case 1:
                # One new node added
                new_nodes = self.add_nodes(extended_state)
                new_edges = np.array([(init_node, new_nodes[0])])

            case _:
                new_nodes = self.add_nodes(extended_state)
                new_edges = np.concatenate(
                    [
                        [(init_node, new_nodes[0])],
                        [(i, i + 1) for i in new_nodes[:-1]],
                    ]
                )
                # Multiple new nodes added
        self.add_edges(new_edges)

        # Step 6. Check if the new state is a solution
        return self.check_solution(new_nodes)
