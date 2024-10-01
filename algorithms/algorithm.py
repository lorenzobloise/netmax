import networkx as nx
from agent import Agent

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
        self.curr_agent_id = curr_agent_id

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")