from src.multi_agent.algorithms.algorithm import Algorithm
import networkx as nx
from src.multi_agent.agent import Agent
import src.multi_agent.competitive_influence_maximization as cim

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

    def __get_marginal_gain_of_u__(self, graph, agents, u):
        """
        :return: marginal gain of node u
        """
        agents[self.curr_agent_id].seed = agents[self.curr_agent_id].seed + [u]
        spread_before = agents[self.curr_agent_id].spread
        curr_marg_gain = cim.simulation(graph, self.diff_model, agents, self.r)[self.agents[self.curr_agent_id].name] - spread_before
        agents[self.curr_agent_id].seed = agents[self.curr_agent_id].seed[:-1]
        return curr_marg_gain