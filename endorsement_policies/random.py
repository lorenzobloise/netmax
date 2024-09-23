from endorsement_policies.endorsement_policy import EndorsementPolicy
import random

class Random(EndorsementPolicy):

    name = "random"

    def __init__(self, graph):
        super().__init__(graph)

    def choose_agent(self, node, graph):
        return random.choice(list(graph.nodes[node]['contacted_by']))