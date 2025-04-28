from .task import BaseTask
from .env import BaseEnv


class BasePlanner:
    def __init__(self, task: BaseTask, env: BaseEnv):
        self.task = task

    def plan(self, state):
        pass
