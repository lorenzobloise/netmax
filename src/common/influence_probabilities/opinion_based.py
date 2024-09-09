from src.common.influence_probabilities.influence_probability import InfluenceProbability
import networkx as nx
import numpy as np

class OpinionBased(InfluenceProbability):

    name = 'opinion'

    def __init__(self, b=0.01):
        super().__init__()
        self.b = b
        self.k = (1-self.b)/2 # 2 is the maximum of the sums of the similarities (both similarities are 1 in this case)

    def __cosine_similarity__(self, vect1, vect2):
        return vect1@vect2.T / (np.linalg.norm(vect1) * np.linalg.norm(vect2))

    def get_probability(self, graph, u, v):
        opinion1 = graph.nodes[u]['opinion']
        opinion2 = graph.nodes[v]['opinion']
        return self.b + self.k * ((1 / graph.out_degree(u)) * nx.simrank_similarity(graph, u, v) + self.__cosine_similarity__(opinion1, opinion2))