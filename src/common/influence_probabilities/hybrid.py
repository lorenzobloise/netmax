from src.common.influence_probabilities.influence_probability import InfluenceProbability
import random

class Hybrid(InfluenceProbability):
    """
    Paper: Gursoy et al. - "Influence Maximization in Social Networks Under Deterministic Linear Threshold Model"
    """

    name = 'hybrid'

    def __init__(self):
        super().__init__()

    def get_probability(self, graph, u, v):
        # Get average global degree
        avg_degree = sum(graph.degree().values()) / len(graph.nodes)
        w1 = 1 / avg_degree
        # Obtain w2 from the product between w1 and the value of a distribution
        p = random.uniform(0.75,1)
        w2 = w1 * p
        # Calculate the final probability as the geometric mean of w1 and w2
        return (w1 * w2) ** 0.5