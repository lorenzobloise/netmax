from netmax.endorsement_policies.endorsement_policy import EndorsementPolicy
import random

class Random(EndorsementPolicy):
    """
    Nodes choose the agent to endorse uniformly at random.
    """

    name = "random"

    def __init__(self, graph):
        super().__init__(graph)

    def choose_agent(self, node, graph):
        return random.choice(list(graph.nodes[node]['contacted_by']))