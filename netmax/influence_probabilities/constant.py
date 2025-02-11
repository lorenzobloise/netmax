from netmax.influence_probabilities.influence_probability import InfluenceProbability

class Constant(InfluenceProbability):
    """
    Sets the influence probability at a constant value (0.1) for each edge in the graph.
    """

    name = 'constant'

    def __init__(self):
        super().__init__()

    def get_probability(self, graph, u, v):
        """
        :param graph: the input graph.
        :param u: the source node.
        :param v: the target node.
        :return: the inferred influence probability on the edge (u,v).
        """
        return 0.1