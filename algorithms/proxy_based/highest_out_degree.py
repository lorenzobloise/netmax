from networkx import DiGraph
from algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm
import influence_maximization as im

class HighestOutDegree(ProxyBasedAlgorithm):
    """
    The Highest Out-Degree algorithm selects nodes based on their out-degree,
    which is the number of edges directed outwards from a node.
    The idea is that nodes with higher out-degree have more influence over other nodes in the network.
    """
    name = 'outdeg'

    def __init__(self, graph: DiGraph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.out_deg_ranking = None

    def run(self):
        if self.out_deg_ranking is None:
            self.out_deg_ranking = sorted(im.inactive_nodes(self.graph), key=lambda node: self.graph.out_degree(node))
        seed_set = []
        for _ in range(self.budget):
            seed_set.append(self.out_deg_ranking.pop())
        return seed_set, {a.name: 0 for a in self.agents}