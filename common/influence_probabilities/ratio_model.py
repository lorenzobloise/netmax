from common.influence_probabilities.influence_probability import InfluenceProbability

class RatioModel(InfluenceProbability):

    name = 'ratio'

    def __init__(self):
        super().__init__()

    def get_probability(self, graph, u, v):
        return 1 / graph.in_degree(v)