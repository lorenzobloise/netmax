from endorsement_policies.endorsement_policy import EndorsementPolicy
import influence_maximization as im
import networkx as nx

class Community(EndorsementPolicy):

    name = "community"

    def __init__(self, graph):
        super().__init__(graph)
        self.communities = nx.community.louvain_communities(graph)

    def __find_community__(self, node):
        for community in self.communities:
            if node in community:
                return community
        return None

    def choose_agent(self, node, graph):
        community = self.__find_community__(node)
        scores = dict()
        for u in community:
            if im.is_active(graph, u):
                agent = graph.nodes[u]['agent']
                scores[agent] = scores.get(agent, 0) + 1
        return max(scores, key=scores.get)