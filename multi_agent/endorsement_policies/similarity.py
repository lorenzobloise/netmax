from multi_agent.endorsement_policies.endorsement_policy import EndorsementPolicy
import multi_agent.competitive_influence_maximization as cim
import networkx as nx

class Similarity(EndorsementPolicy):

    name = "similarity_endorsement"

    def __init__(self, graph):
        super().__init__(graph)
        self.similarity = nx.simrank_similarity(graph)

    def choose_agent(self, node, graph):
        scores = dict()
        for neighbor in graph[node]:
            # Check if the neighbor is already activated
            if cim.is_active(graph, neighbor):
                # Compute similarity
                agent = graph.nodes[neighbor]['agent']
                scores[agent] = scores.get(agent, 0) + self.similarity[node][neighbor]
        # Choose the agent with the most votes
        return max(scores, key=scores.get)