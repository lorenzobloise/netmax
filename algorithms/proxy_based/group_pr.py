import copy
import networkx as nx
from heapdict import heapdict
from algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm

class Group_PR(ProxyBasedAlgorithm):
    """
    Paper: Liu et al. - "Influence Maximization over Large-Scale Social Networks A Bounded Linear Approach"

    """
    name = 'group_pr'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.d = 0.85
        self.inverted_graph = self.graph.reverse(copy=True)
        self.influencee = list(self.graph.nodes) # Nodes that can be influenced for each agent
        self.delta_dict = None # Dictionary of heaps which store the value of delta for each node and for each agent

    def __initialize_delta_dict__(self):
        # Compute influence-PageRank vector
        personalization = {u: 1 / len(self.influencee) for u in self.influencee}
        fPR = nx.pagerank(self.inverted_graph, alpha=self.d, personalization=personalization, weight='p')
        curr_delta_dict = heapdict()
        for s in self.graph.nodes():
            curr_delta_dict[s] = - ((len(self.influencee) / (1 - self.d)) * fPR[s])
        self.delta_dict = {a.id: copy.deepcopy(curr_delta_dict) for a in self.agents}

    def __in_some_seed_set__(self, v, agents):
        for a in agents:
            if v in a.seed:
                return True
        return False

    def __remove_node_from_heaps__(self, v):
        for a in self.agents:
            del self.delta_dict[a.id][v]

    def __get_delta_bound__(self, seed_set, s):
        if len(self.influencee) == 0:
            fPR = nx.pagerank(self.inverted_graph, alpha=self.d, weight='p')
        else:
            personalization = {u: 1 / len(self.influencee) for u in self.influencee}
            fPR = nx.pagerank(self.inverted_graph, alpha=self.d, personalization=personalization, weight='p')
        delta_s = fPR[s]
        for j in seed_set:
            p_js = self.graph.edges[j, s]['p'] if self.graph.has_edge(j, s) else 0
            p_sj = self.graph.edges[s, j]['p'] if self.graph.has_edge(s, j) else 0
            delta_s = delta_s - self.d * p_js * fPR[s] - self.d * p_sj * fPR[j]
        return delta_s * (len(self.influencee) / (1 - self.d))

    def run(self):
        self.__update_active_nodes__()
        if self.delta_dict is None:
            self.__initialize_delta_dict__()
        agents_copy = copy.deepcopy(self.agents)
        # Take the node which has the maximum value of delta
        added_nodes = 0
        while added_nodes < self.budget:
            s, neg_delta = self.delta_dict[self.curr_agent_id].popitem()
            self.delta_dict[self.curr_agent_id][s] = -self.__get_delta_bound__(agents_copy[self.curr_agent_id].seed, s)
            if s == self.delta_dict[self.curr_agent_id].peekitem()[0]:
                s_max, _ = self.delta_dict[self.curr_agent_id].peekitem()
                agents_copy[self.curr_agent_id].seed.append(s_max)
                self.__remove_node_from_heaps__(s_max)
                self.influencee.remove(s_max)
                added_nodes += 1
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: 0 for a in self.agents}