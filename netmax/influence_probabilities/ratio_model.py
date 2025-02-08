from netmax.influence_probabilities.influence_probability import InfluenceProbability

class RatioModel(InfluenceProbability):
    """
    Assigns the influence probability to the edge (u,v) as 1 divided by the number of in-neighbors of v.
    """

    name = 'ratio'

    def __init__(self):
        super().__init__()

    def get_probability(self, graph, u, v):
        """
        :param graph: the input graph.
        :param u: the source node.
        :param v: the target node.
        :return: the inferred influence probability on the edge (u,v).
        """
        return 1 / graph.in_degree(v)