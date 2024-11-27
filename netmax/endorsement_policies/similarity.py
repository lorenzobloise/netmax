from netmax.endorsement_policies.endorsement_policy import EndorsementPolicy
from netmax import influence_maximization as im
import networkx as nx

class Similarity(EndorsementPolicy):
    """
    The nodes choose the agent to endorse based on a score computed as follows.
    - For each agent initialize a score to 0
    - Then, for each of their active in and out neighbors, compute the SimRank similarity between the two nodes (the SimRank matrix is computed only once) and sum it to the agent that the neighbor has endorsed
    Finally, choose the agent by picking the one who has the maximum score. So it's like a voting strategy but weighted with the similarities.
    """

    name = "sim_endorsement"

    def __init__(self, graph):
        super().__init__(graph)
        self.similarity = nx.simrank_similarity(graph)

    def choose_agent(self, node, graph):
        scores = dict()
        for neighbor in set(list(graph.predecessors(node))+list(graph.successors(node))):
            # Check if the neighbor is already activated
            if im.is_active(graph, neighbor):
                # Compute similarity
                agent = graph.nodes[neighbor]['agent']
                scores[agent] = scores.get(agent, 0) + self.similarity[node][neighbor]
        # Choose the agent with the most votes
        return max(scores, key=scores.get)