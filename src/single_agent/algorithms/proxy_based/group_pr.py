import networkx as nx
from tqdm import tqdm
from src.single_agent.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm

class Group_PR(ProxyBasedAlgorithm):
    """
    Paper: Liu et al. - "Influence Maximization over Large-Scale Social Networks A Bounded Linear Approach"
    """
    name = 'group-pr'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)
        self.d = 0.85
        self.inverted_graph = self.graph.reverse(copy=True)

    def __get_delta_bound__(self, seed_set, influencee, s):
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
        influencee = list(self.graph.nodes)
        # Compute influence-PageRank vector
        fPR = nx.pagerank(self.inverted_graph, alpha=self.d, weight='p')
        delta_dict = {s: (len(influencee) / (1 - self.d)) * fPR[s] for s in self.graph.nodes}
        progress_bar = tqdm(range(self.budget), desc='Group_PR', position=1, leave=True)
        while len(seed_set) < self.budget:
            # Re-arrange the order of nodes to make delta_s > delta_{s+1}
            delta_dict = dict(sorted(delta_dict.items(), key=lambda item: item[1], reverse=True))
            delta_max, s_max = 0, -1
            for s in [_ for _ in self.graph.nodes if _ not in seed_set]:
                if delta_dict[s] > delta_max:
                    # Compute the real increment delta_s by Linear or by Bound
                    delta_dict[s] = self.__get_delta_bound__(seed_set, influencee, s)
                    if delta_dict[s] > delta_max:
                        delta_max, s_max = delta_dict[s], s
            seed_set.append(s_max)
            progress_bar.update(1)
            influencee.remove(s_max)
            delta_dict[s_max] = 0
        return seed_set
