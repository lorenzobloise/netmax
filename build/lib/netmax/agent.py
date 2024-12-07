import copy

class Agent(object):

    def __init__(self, name: str, budget: int, id: int = -1):
        """
        This class models an agent.
        :param name: The name of the agent.
        :param budget: The budget of the agent.
        :param id: The id of the agent.
        """
        self.name: str = name
        self.budget: int = budget
        self.seed: [int] = []
        self.spread = 0
        self.id: int = id

    def __deepcopy__(self, memodict={}):
        """
        Makes a deep copy of the agent object.
        """
        new_agent = Agent(self.name, self.budget)
        new_agent.seed = copy.deepcopy(self.seed)
        new_agent.spread = self.spread
        new_agent.id = self.id
        return new_agent