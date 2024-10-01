from algorithms.algorithm import Algorithm
import networkx as nx
from agent import Agent
import random
import copy

class SketchBasedAlgorithm(Algorithm):
    """
    Sketch-based algorithms for seed set selection in influence maximization problems improve the theoretical efficiency
    of simulation-based methods while preserving the approximation guarantee. To avoid rerunning the Monte Carlo
    simulations, a number of "sketches" based on the specific diffusion model are pre-computed and exploited to evaluate
    the influence spread.
    """

    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.transposed_graph = self.graph.reverse(copy=True)
        self.diff_model_transposed = self.diff_model.__copy__()

    def __generate_sketch__(self):
        """
        Sample each edge (u,v) from the graph according to its probability p(u,v)
        """
        sketch = nx.DiGraph()
        sketch.add_nodes_from(list(self.graph.nodes))
        for (u, v, attr) in self.graph.edges(data=True):
            r = random.random()
            if r < attr['p']:
                sketch.add_edge(u, v)
        return sketch

    def __generate_random_reverse_reachable_set__(self, random_node):
        agents_copy = copy.deepcopy(self.agents)
        agents_copy[self.curr_agent_id].seed.append(random_node)
        active_set = self.diff_model_transposed.activate(self.transposed_graph, agents_copy)[self.agents[self.curr_agent_id].name]
        return set(active_set)

    def __in_degree_positive_edges__(self, node):
        """
        Custom method definition to handle negative edge weights in signed graphs.
        In this method we compute the in-degree taking into account only the positive in-edges
        """
        in_degree = 0
        for predecessor, _, data in self.graph.in_edges(node, data=True):
            if data.get('p', 0) > 0:
                in_degree += 1
        return in_degree

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")