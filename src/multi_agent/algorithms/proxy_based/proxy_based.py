import networkx as nx
from src.multi_agent.agent import Agent
from src.multi_agent.algorithms.algorithm import Algorithm

class ProxyBasedAlgorithm(Algorithm):
    """
    Proxy-based algorithms for seed set selection use heuristic measures
    to identify influential nodes in a network. These algorithms do not rely on extensive simulations
    but instead use structural properties of the graph to make decisions.
    """
    def __init__(self, graph: nx.DiGraph, agent: Agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)
        self.graph = graph
        self.agent = agent
        self.budget = budget
        self.diff_model = diff_model
        self.r = r

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")