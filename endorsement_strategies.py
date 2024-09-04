import random
import im
import networkx as nx

class EndorsementStrategy:

    def __init__(self, graph):
        return

    def choose_agent(self, node, graph):
        raise NotImplementedError("This method must be implemented by subclasses")


class Random(EndorsementStrategy):

    name = "random"

    def __init__(self, graph):
        super().__init__(graph)
        return

    def choose_agent(self, node, graph):
        return random.choice(graph.nodes[node]['contacted_by'])


class Voting(EndorsementStrategy):

    name = "voting"

    def __init__(self, graph):
        super().__init__(graph)
        return

    def choose_agent(self, node, graph):
        voting = dict()
        for neighbor in graph[node]:
            # Check if the neighbor is already activated
            if im.is_active(graph, neighbor):
                agent_name = graph.nodes[neighbor]['agent']
                voting[agent_name] = voting.get(agent_name, 0) + 1
        # Choose the agent with the most votes
        return max(voting, key=voting.get)


class Similarity(EndorsementStrategy):

    name = "similarity"

    def __init__(self, graph):
        super().__init__(graph)
        self.similarity = nx.simrank_similarity(graph)

    def choose_agent(self, node, graph):
        scores = dict()
        for neighbor in graph[node]:
            # Check if the neighbor is already activated
            if im.is_active(graph, neighbor):
                # Compute similarity
                agent_name = graph.nodes[neighbor]['agent']
                scores[agent_name] = scores.get(agent_name, 0) + self.similarity[node][neighbor]
        # Choose the agent with the most votes
        return max(scores, key=scores.get)


class Community(EndorsementStrategy):

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
                agent_name = graph.nodes[u]['agent']
                scores[agent_name] = scores.get(agent_name, 0) + 1
        return max(scores, key=scores.get)