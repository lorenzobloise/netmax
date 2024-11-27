from netmax.influence_probabilities.influence_probability import InfluenceProbability
import random

class Uniform(InfluenceProbability):
    """
    Samples the influence probability from a uniform distribution between 0.01 and 0.1.
    """

    name = 'uniform'

    def __init__(self):
        super().__init__()

    def get_probability(self, graph, u, v):
        return random.uniform(0.01,0.1)