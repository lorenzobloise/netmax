import networkx as nx
from heapdict import heapdict
import numpy as np
from tqdm import tqdm
import im

class ProxyBasedAlgorithm:
    """
    Proxy-based algorithms for seed set selection in influence maximization problems use heuristic measures
    to identify influential nodes in a network. These algorithms do not rely on extensive simulations
    but instead use structural properties of the graph to make decisions.
    """

    def __init__(self, graph, agent, budget, diff_model, r):
        self.graph = graph
        self.agent = agent
        self.budget = budget

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")


class HighestOutDegree(ProxyBasedAlgorithm):
    """
    The Highest Out-Degree algorithm selects nodes based on their out-degree,
    which is the number of edges directed outwards from a node.
    The idea is that nodes with higher out-degree have more influence over other nodes in the network.
    """

    name = 'outdeg'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        out_deg_ranking = sorted(self.graph.nodes, key=lambda node: self.graph.out_degree(node), reverse=True)
        seed_set = out_deg_ranking[:self.budget]
        return seed_set


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
        for _ in tqdm(range(self.budget), desc="DegDis",position=1, leave=True):
            u, _ = dd.popitem()
            seed_set.append(u)
            for v in self.graph[u]:
                if v not in seed_set:
                    # Get label p of edge u v
                    if v not in p:  # If v hasn't been reached yet
                        p[v] = self.graph.edges[u, v]['p']
                    elif p[v] < self.graph.edges[u, v]['p']:
                        p[v] = self.graph.edges[u, v]['p']
                    t[v] += 1  # Increase number of selected neighbors
                    score = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * p[v]
                    dd[v] = -score
        return seed_set


class UBound(ProxyBasedAlgorithm):
    """
    Paper: Zhou et al. - "UBLF: An Upper Bound Based Approach to Discover Influential Nodes in Social Networks"
    """

    name = 'ubound'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def get_propagation_probability_matrix(self, graph):
        """
        :param graph:
        :return: PP: propagation probability matrix
        """
        num_nodes = graph.number_of_nodes()
        PP = np.zeros((num_nodes, num_nodes))
        for (u, v, data) in graph.edges(data=True):
            PP[u][v] = data['p']
        return PP

    def get_delta_appr(self, graph):
        PP = self.get_propagation_probability_matrix(graph)
        one_vector = np.ones((graph.number_of_nodes(), 1))
        a_t = [np.dot(PP, one_vector)]
        progress_bar = tqdm()
        while True:
            a_t.append(np.dot(PP, a_t[-1]))
            if np.linalg.norm(a_t[-1], ord=1) < 10 ** (-6):
                break
            progress_bar.update(1)
        return np.sum(a_t, axis=0)

    def get_delta_exact(self, graph):
        PP = self.get_propagation_probability_matrix(graph)
        E = np.eye(PP.shape[0], PP.shape[1])
        delta = np.dot(np.linalg.inv(E - PP), np.ones((graph.number_of_nodes(), 1)))
        return delta

    def run(self):
        delta = self.get_delta_exact(self.graph)  # Vector N x 1
        delta_dict = {i: delta[i] for i in range(len(delta))}
        delta_dict = dict(sorted(delta_dict.items(), key=lambda item: item[1], reverse=True))
        seed_set = list(delta_dict.keys())[:self.budget]
        return seed_set


class Group_PR(ProxyBasedAlgorithm):
    """
    Paper: Liu et al. - "Influence Maximization over Large-Scale Social Networks A Bounded Linear Approach"
    """

    name = 'group-pr'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)
        self.d = 0.85
        self.inverted_graph = im.invert_edges(self.graph)

    def get_delta_bound(self, seed_set, influencee, s):
        if len(influencee) == 0:
            fPR = nx.pagerank(self.inverted_graph, alpha=self.d, weight='p')
        else:
            personalization = {u: 1 / len(influencee) for u in influencee}
            fPR = nx.pagerank(self.inverted_graph, alpha=self.d, personalization=personalization, weight='p')
        delta_s = fPR[s]
        for j in seed_set:
            p_js = self.graph.edges[j, s]['p'] if self.graph.has_edge(j, s) else 0
            p_sj = self.graph.edges[s, j]['p'] if self.graph.has_edge(s, j) else 0
            delta_s = delta_s - self.d * p_js * fPR[s] - self.d * p_sj * fPR[j]
        return delta_s * (len(influencee) / (1 - self.d))

    def run(self):
        seed_set = []
        influencee = list(self.graph.nodes)  # In the beginning, seed set is empty, so all nodes can be influenced
        # Compute influence-PageRank vector
        fPR = nx.pagerank(self.inverted_graph, alpha=self.d, weight='p')
        delta_dict = {s: (len(influencee) / (1 - self.d)) * fPR[s] for s in self.graph.nodes}
        progress_bar = tqdm(range(self.budget), desc='Group_PR')
        while len(seed_set) < self.budget:
            # Re-arrange the order of nodes to make delta_s > delta_{s+1}
            delta_dict = dict(sorted(delta_dict.items(), key=lambda item: item[1], reverse=True))
            delta_max, s_max = 0, -1
            for s in [_ for _ in self.graph.nodes if _ not in seed_set]:
                if delta_dict[s] > delta_max:
                    # Compute the real increment delta_s by Linear or by Bound
                    delta_dict[s] = self.get_delta_bound(seed_set, influencee, s)
                    if delta_dict[s] > delta_max:
                        delta_max, s_max = delta_dict[s], s
            seed_set.append(s_max)
            progress_bar.update(1)
            influencee.remove(s_max)
            delta_dict[s_max] = 0
        return seed_set