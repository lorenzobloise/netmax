from src.multi_agent.endorsement_policies.endorsement_policy import EndorsementPolicy
import random

class Random(EndorsementPolicy):

    name = "random"

    def __init__(self, graph):
        super().__init__(graph)
        return

    def choose_agent(self, node, graph):
        return random.choice(graph.nodes[node]['contacted_by'])