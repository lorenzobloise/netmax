from netmax.influence_probabilities.influence_probability import InfluenceProbability
import networkx as nx

class Similarity(InfluenceProbability):
    """
    Assigns the influence probability to the edge (u,v) as the SimRank value between the two nodes.
    The SimRank matrix is calculated only once, at the beginning of the influence probabilities computing.
    """

    name = 'similarity'

    def __init__(self):
        super().__init__()
        self.similarity = None

    def get_probability(self, graph, u, v):
        if self.similarity is None:
            self.similarity = nx.simrank_similarity(graph)
        return self.similarity[u][v]