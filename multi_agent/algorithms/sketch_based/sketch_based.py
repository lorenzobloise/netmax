from multi_agent.algorithms.algorithm import Algorithm
import networkx as nx
from multi_agent.agent import Agent
import random

class SketchBasedAlgorithm(Algorithm):
    """
    Sketch-based algorithms for seed set selection in influence maximization problems improve the theoretical efficiency
    of simulation-based methods while preserving the approximation guarantee. To avoid rerunning the Monte Carlo
    simulations, a number of "sketches" based on the specific diffusion model are pre-computed and exploited to evaluate
    the influence spread.
    """

    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)

    def __generate_sketch__(self):
        """
        Sample each edge (u,v) from the graph according to its probability p(u,v)
        """
        sampled_edges = []
        for (u, v, attr) in self.graph.edges(data=True):
            r = random.random()
            if r < attr['p']:
                sampled_edges.append((u, v))
        sketch = nx.DiGraph()
        sketch.add_nodes_from(list(self.graph.nodes))
        sketch.add_edges_from(sampled_edges)
        return sketch

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")