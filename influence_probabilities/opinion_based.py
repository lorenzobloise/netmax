from influence_probabilities.influence_probability import InfluenceProbability
import networkx as nx
import numpy as np

class OpinionBased(InfluenceProbability):
    """
    Used only in multi-agent setting. This influence probability requires that the nodes are associated with an 'opinion' information about the agents.
    If the graph does not contain such information, set the 'insert_opinion' parameter at True in the Competitive Influence Maximization class.
    """

    name = 'opinion'

    def __init__(self):
        super().__init__()
        self.b = 0.01
        self.k = (1-self.b)/2 # 2 is the maximum of the sums of the similarities (both similarities are 1 in this case)
        self.similarity = None
        self.probability_cache = dict()
        self.opinion_cache = dict()

    def __cosine_similarity__(self, vect1, vect2):
        return np.dot(vect1, vect2) / (np.linalg.norm(vect1) * np.linalg.norm(vect2))

    def get_probability(self, graph, u, v):
        try:
            opinion1 = graph.nodes[u]['opinion']
            opinion2 = graph.nodes[v]['opinion']
            if self.similarity is None:
                self.similarity = nx.simrank_similarity(graph)
        except KeyError:
            raise KeyError('The nodes must have an opinion attribute to use the OpinionBased influence probability.')
        return self.b + self.k * ((1 / graph.out_degree(u)) * self.similarity[u][v] + self.__cosine_similarity__(opinion1, opinion2))

    def update_probability(self, graph, u, agent):
        if u not in self.opinion_cache:
            self.opinion_cache[u] = graph.nodes[u]['opinion']
        num_agents = len(graph.nodes[u]['opinion'])
        graph.nodes[u]['opinion'] = [0 if i != agent.id else 1 for i in range(num_agents)]
        out_edges = graph.out_edges(u, data=True)
        if u not in self.probability_cache:
            self.probability_cache[u] = dict()
        for (_, v, attr) in out_edges:
            if v not in self.probability_cache[u]:
                self.probability_cache[u][v] = attr['p']
            attr['p'] = self.get_probability(graph, u, v)

    def restore_probability(self, graph, u):
        graph.nodes[u]['opinion'] = self.opinion_cache[u]
        for (_, v, attr) in graph.out_edges(u, data=True):
            attr['p'] = self.probability_cache[u][v]