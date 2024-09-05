from src.single_agent.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm

class HighestOutDegree(ProxyBasedAlgorithm):
    """
    The Highest Out-Degree algorithm selects nodes based on their out-degree,
    which is the number of edges directed outwards from a node.
    The idea is that nodes with higher out-degree have more influence over other nodes in the network.
    """

    name = 'outdeg'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        out_deg_ranking = sorted(im.inactive_nodes(self.graph), key=lambda node: self.graph.out_degree(node), reverse=True)
        seed_set = out_deg_ranking[:self.budget]
        return seed_set