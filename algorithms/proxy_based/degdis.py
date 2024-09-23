import copy
from algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm
from heapdict import heapdict

class DegDis(ProxyBasedAlgorithm):
    """
    The Degree Discount heuristic is an improvement over the Highest Out-Degree algorithm.
    It takes into account the influence of already selected nodes
    and adjusts the degree of remaining nodes accordingly.
    Paper: Chen et al. - "Efficient influence maximization in social networks"
    """
    name = 'degdis'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.d = None # Degree of each vertex
        self.t = None # Number of adjacent vertices that are in the seed set
        self.p = None # Influence probabilities
        self.dd = None # Degree discount

    def __initialize_degree_discount__(self):
        self.d = {}
        self.p = {a.id: {} for a in self.agents}
        self.t = {a.id: {} for a in self.agents}
        self.dd = {a.id: heapdict() for a in self.agents}
        for u in self.graph.nodes():
            self.d[u] = self.graph.out_degree(u)
            for a in self.agents:
                self.dd[a.id][u] = -self.d[u]
                self.t[a.id][u] = 0

    def __delete_from_dd__(self, v):
        for a in self.agents:
            del self.dd[a.id][v]

    def __in_some_seed_set__(self, v, agents):
        for a in agents:
            if v in a.seed:
                return True
        return False

    def __compute_node_score__(self, v):
        return self.d[v] - 2 * self.t[self.curr_agent_id][v] - (self.d[v] - self.t[self.curr_agent_id][v]) * self.t[self.curr_agent_id][v] * self.p[self.curr_agent_id][v]

    def run(self):
        # Initialize degrees and degree discounts
        if self.dd is None:
            self.__initialize_degree_discount__()
        # Add vertices to the seed set of the current agent
        agents_copy = copy.deepcopy(self.agents)
        for _ in range(self.budget):
            u, _ = self.dd[self.curr_agent_id].peekitem()
            agents_copy[self.curr_agent_id].seed.append(u)
            self.__delete_from_dd__(u) # Delete u from the degree discount of all agents
            for v in self.graph[u]: # Neighbors of node u
                if not self.__in_some_seed_set__(v, agents_copy): # If the node is not part of any seed set
                    # Get label p of edge (u, v)
                    if v not in self.p[self.curr_agent_id]: # If v hasn't been reached yet
                        self.p[self.curr_agent_id][v] = self.graph.edges[u, v]['p']
                    elif self.p[self.curr_agent_id][v] < self.graph.edges[u, v]['p']:
                        self.p[self.curr_agent_id][v] = self.graph.edges[u, v]['p']
                    self.t[self.curr_agent_id][v] += 1 # Increase the number of selected neighbors
                    score = self.__compute_node_score__(v)
                    self.dd[self.curr_agent_id][v] = -score
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: 0 for a in self.agents}