from heapdict import heapdict
from tqdm import tqdm
from src.single_agent.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm

class DegDis(ProxyBasedAlgorithm):
    """
    The Degree Discount heuristic is an improvement over the Highest Out-Degree algorithm.
    It takes into account the influence of already selected nodes
    and adjusts the degree of remaining nodes accordingly.
    Paper: Chen et al. - "Efficient influence maximization in social networks"
    """

    name = 'degdis'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        seed_set = []
        # Take the degree of each node
        dd = heapdict()  # Degree discount
        d = {}  # Degree of each vertex
        t = {}  # Number of adjacent vertices that are in the seed set
        p = {}  # Influence probabilities
        # Initialize degrees and degree discounts
        for u in self.graph.nodes():
            d[u] = self.graph.out_degree(u)
            dd[u] = -d[u]
            t[u] = 0
        # Add vertices to seed set
        for _ in tqdm(range(self.budget), desc="DegDis", position=1, leave=True):
            u, _ = dd.popitem()
            seed_set.append(u)
            for v in self.graph[u]:
                if v not in seed_set: # If the node is not part of the seed set
                    # Get label p of edge u v
                    if v not in p:  # If v hasn't been reached yet
                        p[v] = self.graph.edges[u, v]['p']
                    elif p[v] < self.graph.edges[u, v]['p']:
                        p[v] = self.graph.edges[u, v]['p']
                    t[v] += 1  # Increase number of selected neighbors
                    score = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * p[v]
                    dd[v] = -score
        return seed_set