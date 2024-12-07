from networkx import DiGraph
from netmax.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm
from netmax import influence_maximization as im


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
        """
        :return: The nodes to add in the seed set of the current agent and the spreads for each agent.
        """
        # This method is necessary since when the input network is signed, the graph of the proxy-based algorithm
        # contains only the trust-edges (see super-class ProxyBasedAlgorithm)
        self.__update_active_nodes__()
        # Compute the out-degrees if not already done
        if self.out_deg_ranking is None:
            self.out_deg_ranking = sorted(im.inactive_nodes(self.graph), key=lambda node: self.graph.out_degree(node))
        # Iteratively, take the nodes with the highest out-degree.
        # Repeats until the budget is fulfilled (in the InfluenceMaximization class, inside the run method, the algorithm
        # is always invoked with the budget parameter set to 1, but we preferred to write the code in a more general way)
        seed_set = []
        for _ in range(self.budget):
            seed_set.append(self.out_deg_ranking.pop())
        # Return the new nodes to add to the seed set and the spread (which is 0 because we didn't do any simulation,
        # in fact this is only a fictional value, since the real spread will be computed at the end of the game)
        return seed_set, {a.name: 0 for a in self.agents}