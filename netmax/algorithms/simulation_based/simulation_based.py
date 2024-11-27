from netmax.algorithms.algorithm import Algorithm
import networkx as nx
from netmax.agent import Agent

class SimulationBasedAlgorithm(Algorithm):
    """
    Simulation-based algorithms for seed set selection in influence maximization problems rely on simulating the spread
    of influence through a network to identify the most influential nodes.
    These algorithms use Monte Carlo simulations to estimate the expected spread of influence for different sets of seed
    nodes. The goal is to select a set of nodes that maximizes the spread of influence within a given budget.
    """

    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")