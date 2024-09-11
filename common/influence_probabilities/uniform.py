from common.influence_probabilities.influence_probability import InfluenceProbability
import random

class Uniform(InfluenceProbability):

    name = 'uniform'

    def __init__(self):
        super().__init__()

    def get_probability(self, graph, u, v):
        return random.uniform(0.01,0.1)