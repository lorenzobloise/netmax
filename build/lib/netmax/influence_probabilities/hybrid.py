from netmax.influence_probabilities.influence_probability import InfluenceProbability
import random

class Hybrid(InfluenceProbability):
    """
    Paper: Gursoy et al. - "Influence Maximization in Social Networks Under Deterministic Linear Threshold Model".
    Assigns the probability to the edge (u,v) the following way. First, it computes two numbers:
    - w1: 1 divided by the average degree of the graph
    - w2: w1 multiplied by a random number sampled from a uniform distribution between 0.75 and 1
    Then it returns the probability as the geometric mean of w1 and w2.
    """

    name = 'hybrid'

    def __init__(self):
        super().__init__()
        self.avg_degree = None

    def get_probability(self, graph, u, v):
        # Get average global degree if not already computed
        if self.avg_degree is None:
            self.avg_degree = sum(dict(graph.degree()).values()) / len(graph.nodes)
        w1 = 1 / self.avg_degree
        # Obtain w2 from the product between w1 and the value of a distribution
        p = random.uniform(0.75,1)
        w2 = w1 * p
        # Calculate the final probability as the geometric mean of w1 and w2
        return (w1 * w2) ** 0.5