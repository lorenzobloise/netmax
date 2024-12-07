from netmax.algorithms.algorithm import Algorithm
import networkx as nx
from netmax.agent import Agent
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
        # In the random Reverse Reachable Set generation, removing each edge with a certain probability was less
        # efficient than running simulations, because of the highly-efficient diffusion model of this framework, so
        # instead we use a transposed graph with all the edges inverted, and we run simulations from random nodes
        self.transposed_graph = self.graph.reverse(copy=True)
        # We also make a deep copy of the diffusion model, and that's needed because before running the simulations
        # the diffusion model needs to preprocess the transposed graph, and the main diffusion model is already
        # working on the original graph
        self.diff_model_transposed = self.diff_model.__copy__()
        self.diff_model_transposed.preprocess_data(self.transposed_graph)

    def __generate_sketch__(self):
        """
        Generate a sketch by sampling each edge (u,v) from the graph according to its probability p(u,v).
        Since a sketch is different from a random Reverse Reachable Set (a sketch has all the nodes, while the
        RR set does not), here is more convenient the edge sampling strategy instead of running a simulation.
        :return: the generated sketch.
        """
        sketch = nx.DiGraph()
        sketch.add_nodes_from(list(self.graph.nodes)) # A sketch has all the nodes in the graph, but not all the edges
        for (u, v, attr) in self.graph.edges(data=True):
            r = random.random()
            if r < attr['p']:
                sketch.add_edge(u, v)
        return sketch

    def __generate_random_reverse_reachable_set__(self, random_node):
        """
        Run a simulation on the transposed graph, from a random node, to generate a random Reverse Reachable Set.
        In the random Reverse Reachable Set generation, removing each edge with a certain probability is less
        efficient than running simulations, because of the highly-efficient diffusion model of this framework, so
        instead we use a transposed graph with all the edges inverted, and we run simulations from random nodes.
        :param random_node: the node from which the simulation starts.
        :return: the generated random Reverse Reachable Set.
        """
        agents_copy = copy.deepcopy(self.agents)
        agents_copy[self.curr_agent_id].seed.append(random_node)
        active_set = self.diff_model_transposed.activate(self.transposed_graph, agents_copy)[self.agents[self.curr_agent_id].name]
        return set(active_set)

    def __in_degree_positive_edges__(self, node):
        """
        Custom method definition to handle negative edge weights in signed graphs.
        In this method we compute the in-degree taking into account only the positive (trusted) in-edges.
        :param node: the node we have to compute the trusted in-degree of.
        :return: the number of trusted in-neighbors of the node.
        """
        in_degree = 0
        for predecessor, _, data in self.graph.in_edges(node, data=True):
            if data.get('p', 0) > 0: # If this in-neighbor is trusted by the node, count it
                in_degree += 1
        return in_degree

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")