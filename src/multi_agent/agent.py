import copy

class Agent(object):

    def __init__(self, name: str, budget: int):
        self.name: str = name
        self.budget: int = budget
        self.seed: [int] = []
        self.spread = 0
        self.id: int = -1

    def __deepcopy__(self, memodict={}):
        new_agent = Agent(self.name, self.budget)
        new_agent.seed = copy.deepcopy(self.seed)
        new_agent.spread = self.spread
        new_agent.id = self.id
        return new_agent
