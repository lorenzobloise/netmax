from endorsement_policies.endorsement_policy import EndorsementPolicy
import influence_maximization as im

class Voting(EndorsementPolicy):

    name = "voting"

    def __init__(self, graph):
        super().__init__(graph)

    def choose_agent(self, node, graph):
        voting = dict()
        for neighbor in set(list(graph.predecessors(node))+list(graph.successors(node))):
            # Check if the neighbor is already activated
            if im.is_active(graph, neighbor):
                agent = graph.nodes[neighbor]['agent']
                voting[agent] = voting.get(agent, 0) + 1
        # Choose the agent with the most votes
        return max(voting, key=voting.get)