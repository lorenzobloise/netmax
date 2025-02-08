from netmax.endorsement_policies.endorsement_policy import EndorsementPolicy
from netmax import influence_maximization as im


class Voting(EndorsementPolicy):
    """
    The nodes choose the agent to endorse by looking at who their in and out neighbors chose and picking the one that occur the most.
    """

    name = "voting"

    def __init__(self, graph):
        super().__init__(graph)

    def choose_agent(self, node, sim_graph):
        """
        The nodes choose the agent to endorse by looking at who their in and out neighbors chose and picking the one that occur the most.
        :param node: the node which has to choose an agent to endorse.
        :param sim_graph: the simulation graph.
        :return: the agent to endorse.
        """
        voting = dict()
        for neighbor in set(list(sim_graph.predecessors(node))+list(sim_graph.successors(node))):
            # Check if the neighbor is already activated
            if im.is_active(sim_graph, neighbor):
                agent = sim_graph.nodes[neighbor]['agent']
                voting[agent] = voting.get(agent, 0) + 1
        # Choose the agent with the most votes
        return max(voting, key=voting.get)