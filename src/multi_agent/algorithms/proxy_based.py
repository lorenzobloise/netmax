import networkx as nx
from src.multi_agent.agent import Agent


class ProxyBasedAlgorithm:
    """
    Proxy-based algorithms for seed set selection use heuristic measures
    to identify influential nodes in a network. These algorithms do not rely on extensive simulations
    but instead use structural properties of the graph to make decisions.
    """
    def __init__(self, graph:nx.DiGraph, agent:Agent,budget, diff_model, r):
        self.graph = graph
        self.agent = agent
        self.budget = budget
        self.diff_model = diff_model
        self.r = r

    def set_graph(self, graph):
        self.graph = graph

    def set_agent(self, agent):
        self.agent = agent

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")

class HighestOutDegree(ProxyBasedAlgorithm):
    """
    This algorithm selects the nodes with the highest out-degree in the network.
    """
    name = 'outdeg'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        # TODO
        pass

class DegDis(ProxyBasedAlgorithm):
    """
    The Degree Discount heuristic is an improvement over the Highest Out-Degree algorithm.
    It takes into account the influence of already selected nodes
    and adjusts the degree of remaining nodes accordingly.
    Paper: Chen et al. - "Efficient influence maximization in social networks"
    """
    name = 'degdis'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        # TODO
        pass

class UBound(ProxyBasedAlgorithm):
    """
    Paper: Zhou et al. - "UBLF: An Upper Bound Based Approach to Discover Influential Nodes in Social Networks"
    The UBound algorithm doesn't work if the graph has cycles.
    """
    name = 'ubound'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        # TODO
        pass

class Group_PR(ProxyBasedAlgorithm):
    """
    Paper: Liu et al. - "Influence Maximization over Large-Scale Social Networks A Bounded Linear Approach"

    """
    name = 'group_pr'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        # TODO
        pass

class SimPath(ProxyBasedAlgorithm):
    """
    Can only be used under Linear Threshold diffusion model.
    Paper: Goyal et al. - "SimPath: An Efficient Algorithm for
    """
    name = 'simpath'

    class Node(object):

        def __init__(self, node, spd=0, spd_induced=0, flag=False):
            self.node = node
            self.spd = 0
            self.spd_induced = 0
            self.flag = flag

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)
        if diff_model.name != 'lt':
            raise ValueError(f"SimPath can only be executed under Linear Threshold diffusion model (set the argument at 'lt')")
        self.lookahead = 5
        self.eta = 0.001


    def run(self):
        # TODO
        pass
