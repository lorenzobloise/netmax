from influence_probabilities.influence_probability import InfluenceProbability

class Constant(InfluenceProbability):

    name = 'constant'

    def __init__(self):
        super().__init__()

    def get_probability(self, graph, u, v):
        return 0.1