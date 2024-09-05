from src.multi_agent.endorsement_policies.endorsement_policy import EndorsementPolicy
import src.multi_agent.competitive_influence_maximization as cim

class Voting(EndorsementPolicy):

    name = "voting"

    def __init__(self, graph):
        super().__init__(graph)
        return

    def choose_agent(self, node, graph):
        voting = dict()
        for neighbor in graph[node]:
            # Check if the neighbor is already activated
            if cim.is_active(graph, neighbor):
                agent_name = graph.nodes[neighbor]['agent']
                voting[agent_name] = voting.get(agent_name, 0) + 1
        # Choose the agent with the most votes
        return max(voting, key=voting.get)