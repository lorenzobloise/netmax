from src.common.influence_probabilities.influence_probability import InfluenceProbability
import networkx as nx

class Similarity(InfluenceProbability):

    name = 'similarity'

    def __init__(self):
        super().__init__()
        self.similarity = None

    def get_probability(self, graph, u, v):
        if self.similarity is None:
            self.similarity = nx.simrank_similarity(graph)
        return self.similarity[u][v]