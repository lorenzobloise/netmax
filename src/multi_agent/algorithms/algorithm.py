import networkx as nx
from src.multi_agent.agent import Agent

class Algorithm:

    def __init__(self, graph: nx.DiGraph, agent: Agent, budget, diff_model, r):
        self.graph = graph
        self.agent = agent
        self.budget = budget
        self.diff_model = diff_model
        self.r = r

    def set_graph(self, graph):
        self.graph = graph

    def set_agent(self, agent):
        self.agent = agent

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")