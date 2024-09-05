from src.common.influence_probabilities.influence_probability import InfluenceProbability
import networkx as nx

class Similarity(InfluenceProbability):

    name = 'similarity'

    def __init__(self):
        super().__init__()

    def get_probability(self, graph, u, v):
        return nx.simrank_similarity(graph, u, v)