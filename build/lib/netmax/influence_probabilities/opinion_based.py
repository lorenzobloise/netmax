from netmax.influence_probabilities.influence_probability import InfluenceProbability
import networkx as nx
import numpy as np

class OpinionBased(InfluenceProbability):
    """
    Used only in multi-agent setting. This influence probability requires that the nodes are associated with an 'opinion' information about the agents.
    If the graph does not contain such information, set the 'insert_opinion' parameter at True in the Competitive Influence Maximization class.
    Given:
    - A parameter b = 0.01, which is a constant to make the minimum value different from 0
    - A parameter k = (1 - b) / 2 = 0.495, where 2 at the denominator is the maximum of the sums of the similarities (both similarities are 1 in this case), used as a normalization constant
    - The SimRank matrix (computed only once)
    - The opinion vectors of the nodes
    The influence probability of the edge (u, v) is obtained the following way:
    b + k * ( ( 1 / out_degree(u) ) * SimRank(u, v) + cosine_similarity( opinion(u), opinion(v) ) )
    Because inside the parenthesis:
    - The first addend is ( 1 / out_degree(u) ) * SimRank(u, v) and can be at maximum 1, when the node u has only one neighbor (v) and the SimRank similarity between u and v is 1 (it happens only when u and v are the same node, but mathematically it works)
    - The second addend is cosine_similarity( opinion(u), opinion(v) ) and can be at maximum 1, when the two opinion vectors are exactly the same
    So their sum can be at maximum 2. We multiply it by k = 0.495 and obtain a maximum of 0.99, then add b = 0.01 and obtain a total maximum of 1.
    Instead, when both addends are 0, the minimum value is b = 0.01.
    """

    name = 'opinion'

    def __init__(self):
        super().__init__()
        self.b = 0.01
        self.k = (1-self.b)/2 # 2 is the maximum of the sums of the similarities (both similarities are 1 in this case)
        self.similarity = None
        # These two caches are used to store probability and opinion values in the dynamic probability context, so when
        # opinions dynamically change and, as a consequence, also the probabilities change, we have to store the
        # previous values to restore the old ones at the end of the simulation
        self.probability_cache = dict()
        self.opinion_cache = dict()

    def __cosine_similarity__(self, vect1, vect2):
        return np.dot(vect1, vect2) / (np.linalg.norm(vect1) * np.linalg.norm(vect2))

    def get_probability(self, graph, u, v):
        try:
            # Get the opinion vectors of the two nodes
            opinion1 = graph.nodes[u]['opinion']
            opinion2 = graph.nodes[v]['opinion']
            # Compute the SimRank similarity, if not already computed
            if self.similarity is None:
                self.similarity = nx.simrank_similarity(graph)
        except KeyError:
            # Handle the case in which the nodes do not have the 'opinion' attribute
            raise KeyError('The nodes must have an opinion attribute to use the OpinionBased influence probability.')
        # Compute the opinion-based formula
        return self.b + self.k * ((1 / graph.out_degree(u)) * self.similarity[u][v] + self.__cosine_similarity__(opinion1, opinion2))

    def update_probability(self, graph, u, agent):
        """
        In this scenario, the node u has been influenced by an agent, so its old opinion has to be stored inside the cache
        and its new opinion is 0 for all the agents except the one he endorses
        """
        # Store the old opinion of the node u
        if u not in self.opinion_cache:
            self.opinion_cache[u] = graph.nodes[u]['opinion']
        # Set the new opinion
        num_agents = len(graph.nodes[u]['opinion'])
        graph.nodes[u]['opinion'] = [0 if i != agent.id else 1 for i in range(num_agents)]
        # For each out edge, update the influence probability according to the opinion-based model
        out_edges = graph.out_edges(u, data=True)
        if u not in self.probability_cache:
            self.probability_cache[u] = dict()
        for (_, v, attr) in out_edges:
            # Store the old influence probability of the edge (u,v)
            if v not in self.probability_cache[u]:
                self.probability_cache[u][v] = attr['p']
            # Update the influence probability
            attr['p'] = self.get_probability(graph, u, v)

    def restore_probability(self, graph, u):
        """
        In this scenario, the node u has been deactivated (it happens either when a simulation has ended or during a dynamic diffusion model simulation),
        so we have to restore both its opinion and the influence probability for each of its out edges.
        """
        # Restore the opinion
        graph.nodes[u]['opinion'] = self.opinion_cache[u]
        # Restore the influence probability for each of its out edges
        for (_, v, attr) in graph.out_edges(u, data=True):
            attr['p'] = self.probability_cache[u][v]