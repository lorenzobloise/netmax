import networkx as nx
from netmax.agent import Agent
from netmax.algorithms.algorithm import Algorithm
from netmax import influence_maximization as im


class ProxyBasedAlgorithm(Algorithm):
    """
    Proxy-based algorithms for seed set selection use heuristic measures
    to identify influential nodes in a network. These algorithms do not rely on extensive simulations
    but instead use structural properties of the graph to make decisions.
    """
    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        # With a signed network, the heuristic must be computed by taking into account only the
        # trusted edges, thus discarding the untrusted ones. That's because, in a signed setting, nodes
        # can only be activated by their trusted in-neighbors. For example, with the out-degree heuristic,
        # picking the node with the highest out-degree does not have any sense if all its edges are
        # negative, because that node can't influence anyone
        if im.graph_is_signed(graph):
            self.graph, _ = im.build_trust_and_distrust_graphs(self.graph, verbose=True)

    def __update_active_nodes__(self):
        """
        This method is necessary since when the input network is signed, the graph of the proxy-based algorithm
        contains only the trust-edges. In this case, the attribute 'graph' of the InfluenceMaximization object is different
        from the one in the algorithm, thus whenever we activate some node inside the InfluenceMaximization class,
        we have to report these activations on the trust graph of the proxy-based algorithm
        """
        if not im.graph_is_signed(self.graph):
            # If the graph is not signed, there is no need to update the active nodes (the graph is the same)
            return
        for a in self.agents:
            for node in a.seed:
                if not im.is_active(node, self.graph):
                    im.activate_node(self.graph, node, a)

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")