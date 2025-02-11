from netmax.endorsement_policies.endorsement_policy import EndorsementPolicy
from netmax import influence_maximization as im
import networkx as nx

class Community(EndorsementPolicy):
    """
    The nodes choose the agent to endorse by applying a voting strategy extended not only to their neighbors but to the entire community they're part of.
    The communities are found only once, at the beginning, by applying Louvain algorithm.
    """

    name = "community"

    def __init__(self, graph):
        super().__init__(graph)
        self.communities = nx.community.louvain_communities(graph)

    def __find_community__(self, node):
        """
        Finds the community which the node belongs to.
        :param node: The node to find the community of.
        :return: The community which the node belongs to.
        """
        for community in self.communities:
            if node in community:
                return community
        return None

    def choose_agent(self, node, sim_graph):
        """
        The nodes choose the agent to endorse by applying a voting strategy extended not only to their neighbors but to the entire community they're part of.
        The communities are found only once, at the beginning, by applying Louvain algorithm.
        :param node: the node which has to choose an agent to endorse.
        :param sim_graph: the simulation graph.
        :return: the agent to endorse.
        """
        # Find the community this node is part of
        community = self.__find_community__(node)
        scores = dict()
        for u in community:
            # Check if this node is active
            if im.is_active(sim_graph, u):
                agent = sim_graph.nodes[u]['agent']
                scores[agent] = scores.get(agent, 0) + 1
        return max(scores, key=scores.get)