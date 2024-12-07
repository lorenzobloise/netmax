import copy
from netmax.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm
from heapdict import heapdict

class DegDis(ProxyBasedAlgorithm):
    """
    Paper: Chen et al. - "Efficient Influence Maximization in Social Networks".
    The Degree Discount heuristic is an improvement over the Highest Out-Degree algorithm. It takes into account the
    influence of already selected nodes and adjusts the degree of remaining nodes accordingly.
    """

    name = 'degdis'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.d = None
        self.t = None
        self.p = None
        self.dd = None

    def __initialize_degree_discount__(self):
        """
        Initializes all the data structures needed for the algorithm. Most of them are agent-dependant, so every agent
        has its own version of the data structure.
        """
        # Influence probabilities for every agent, dictionary of dictionaries <agent: <vertex: influence_probability>>,
        # where we compute this probability as the highest edge label among the in-edges of the vertex, instead
        # the author of the paper sets it as a fixed value (for example 0.01)
        self.p = {a.id: {} for a in self.agents}
        self.t = {a.id: {} for a in self.agents} # Number of adjacent vertices that are in the seed set,
                                                # dictionary of dictionaries <agent: <vertex: adjacent_vertices_in_ss>>
        self.dd = {a.id: heapdict() for a in self.agents} # Degree discount heuristic, dictionary <agent: heapdict>
        self.d = {}  # Degree of each vertex, dictionary <vertex: degree>
        # Build the node degrees
        for u in self.graph.nodes():
            self.d[u] = self.graph.out_degree(u)
            # Initialize the heuristic value as the current degree (negative because of the min-heap),
            # and the number of adjacent vertices that are in the seed set (at this moment 0 of course)
            for a in self.agents:
                self.dd[a.id][u] = -self.d[u]
                self.t[a.id][u] = 0

    def __delete_from_dd__(self, v):
        """
        Removes the node v from the degree discount dictionary.
        :param v: The node to remove.
        """
        for a in self.agents:
            del self.dd[a.id][v]

    def __compute_node_score__(self, v):
        """
        :return: the score of the degree discount heuristic for the node v, as shown in the paper. Only difference is
        that the paper works with fixed-value influence probabilities, while we extend this considering different
        probability values by taking the highest edge label among the in-edges of the vertex.
        """
        return self.d[v] - 2 * self.t[self.curr_agent_id][v] - (self.d[v] - self.t[self.curr_agent_id][v]) * self.t[self.curr_agent_id][v] * self.p[self.curr_agent_id][v]

    def run(self):
        """
        :return: The nodes to add in the seed set of the current agent and the spreads for each agent.
        """
        # This method is necessary since when the input network is signed, the graph of the proxy-based algorithm
        # contains only the trust-edges (see super-class ProxyBasedAlgorithm)
        self.__update_active_nodes__()
        # Initialize degrees and degree discounts if it's the first turn of the first round
        if self.dd is None:
            self.__initialize_degree_discount__()
        # Add vertices to the seed set of the current agent
        agents_copy = copy.deepcopy(self.agents)
        # Repeats until the budget is fulfilled (in the InfluenceMaximization class, inside the run method, the algorithm
        # is always invoked with the budget parameter set to 1, but we preferred to write the code in a more general way)
        for _ in range(self.budget):
            # Select the node with the maximum value of the degree discount heuristic
            u, _ = self.dd[self.curr_agent_id].peekitem()
            agents_copy[self.curr_agent_id].seed.append(u) # Add it into the seed set of the current agent
            self.__delete_from_dd__(u) # Delete u from the degree discount of all agents
            for v in self.graph[u]: # Neighbors of node u
                if not self.__in_some_seed_set__(v, agents_copy): # If the node is not part of any seed set
                    # Compute influence probability of node v as the maximum edge label
                    # among his in-edges (different from the paper)
                    if v not in self.p[self.curr_agent_id]: # If v hasn't been reached yet
                        self.p[self.curr_agent_id][v] = self.graph.edges[u, v]['p']
                    elif self.p[self.curr_agent_id][v] < self.graph.edges[u, v]['p']:
                        self.p[self.curr_agent_id][v] = self.graph.edges[u, v]['p']
                    self.t[self.curr_agent_id][v] += 1 # Increase the number of selected neighbors
                    score = self.__compute_node_score__(v) # Compute the degree-discount heuristic of node v
                    self.dd[self.curr_agent_id][v] = -score
        # Return the new nodes to add to the seed set and the spread (which is 0 because we didn't do any simulation,
        # in fact this is only a fictional value, since the real spread will be computed at the end of the game)
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: 0 for a in self.agents}