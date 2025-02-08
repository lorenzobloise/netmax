from netmax.endorsement_policies.endorsement_policy import EndorsementPolicy
import random

class Random(EndorsementPolicy):
    """
    Nodes choose the agent to endorse uniformly at random.
    """

    name = "random"

    def __init__(self, graph):
        super().__init__(graph)

    def choose_agent(self, node, sim_graph):
        """
        The node chooses the agent to endorse uniformly at random.
        :param node: the node which has to choose an agent to endorse.
        :param sim_graph: the simulation graph.
        :return: the agent to endorse.
        """
        return random.choice(list(sim_graph.nodes[node]['contacted_by']))