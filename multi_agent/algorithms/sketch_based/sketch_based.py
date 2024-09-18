from multi_agent.algorithms.algorithm import Algorithm
import networkx as nx
from multi_agent.agent import Agent

class SketchBasedAlgorithm(Algorithm):
    """
    Sketch-based algorithms for seed set selection in influence maximization problems improve the theoretical efficiency
    of simulation-based methods while preserving the approximation guarantee. To avoid rerunning the Monte Carlo
    simulations, a number of "sketches" based on the specific diffusion model are pre-computed and exploited to evaluate
    the influence spread.
    """

    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")