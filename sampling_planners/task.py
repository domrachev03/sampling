class BaseTask:
    def __init__(self, name: str):
        self.name = name

    def eval(self, state):
        # Returns heuristic value of the state w.r.t. goal
        pass
