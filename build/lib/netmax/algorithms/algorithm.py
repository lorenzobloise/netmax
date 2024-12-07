import networkx as nx
from netmax.agent import Agent

class Algorithm:

    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        """
        :param graph: networkx DiGraph
        :param agents: list of Agent
        :param curr_agent_id: int - index of the current agent
        :param budget: int - budget of the current agent
        :param diff_model: str - diffusion model
        :param r: float - discount factor
        """
        self.graph = graph
        self.agents = agents
        self.curr_agent_id = curr_agent_id
        self.budget = budget
        self.diff_model = diff_model
        self.r = r

    def set_curr_agent(self, curr_agent_id):
        """
        Sets the current agent as the one passed.
        :param curr_agent_id: index of the current agent.
        """
        self.curr_agent_id = curr_agent_id

    def __in_some_seed_set__(self, v, agents):
        """
        Checks if a node is in some seed set.
        :param v: the node to check.
        :param agents: the 'agents' dictionary, which contain all the seed sets.
        :return: True if the node is in some seed set, False otherwise.
        """
        for a in agents:
            if v in a.seed:
                return True
        return False

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")