import copy
import networkx as nx
from heapdict import heapdict
from netmax.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm

class Group_PR(ProxyBasedAlgorithm):
    """
    Paper: Liu et al. - "Influence Maximization over Large-Scale Social Networks A Bounded Linear Approach".
    Group-PageRank starts from the fact that PageRank as un upper bound to the influence of single nodes under
    linear influence processes (and it's called influence-PageRank), and extends this concept to compute the
    influence of groups of nodes via the so-called Group-PageRank. Then it plugs this heuristic into a linear
    framework to maximize the influence spread.
    """

    name = 'group_pr'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.d = 0.85 # Parameter of PageRank
        # PageRank works with incoming links, but the influence propagation has only sense considering outgoing links,
        # so we use the inverted graph to compute PageRank
        self.inverted_graph = self.graph.reverse(copy=True)
        self.influencee = list(self.graph.nodes) # Nodes that can be influenced for each agent
        self.delta_dict = None # Dictionary of heaps which store the value of delta for each node and for each agent

    def __initialize_delta_dict__(self):
        """
        Initialize the dictionary of heaps with the same heap for each agent as in the beginning the delta value
        is the same.
        """
        # Compute influence-PageRank with a bias towards the nodes that can be influenced
        personalization = {u: 1 / len(self.influencee) for u in self.influencee}
        fPR = nx.pagerank(self.inverted_graph, alpha=self.d, personalization=personalization, weight='p')
        curr_delta_dict = heapdict()
        for s in self.graph.nodes():
            # Formula in the paper, negative because we have to insert in the heap which orders in descending order
            curr_delta_dict[s] = - ((len(self.influencee) / (1 - self.d)) * fPR[s])
        self.delta_dict = {a.id: copy.deepcopy(curr_delta_dict) for a in self.agents}

    def __remove_node_from_heaps__(self, v):
        """
        Removes a node from all heaps.
        :param v: the node to remove.
        """
        for a in self.agents:
            del self.delta_dict[a.id][v]

    def __get_delta_bound__(self, seed_set, s):
        """
        Method used to update the entries of the delta dictionary. In the paper there are two ways to do so: a linear
        approach or a bound approach. We chose to implement the bound approach with this method.
        :param seed_set: the seed set.
        :param s: the node which delta value has to be computed.
        :return: the value of delta for the node s.
        """
        # If no node can be influenced, compute the influence-PageRank with the default personalization vector
        if len(self.influencee) == 0:
            fPR = nx.pagerank(self.inverted_graph, alpha=self.d, weight='p')
        # Otherwise compute the influence-PageRank with a bias towards the nodes that can be influenced
        else:
            personalization = {u: 1 / len(self.influencee) for u in self.influencee}
            fPR = nx.pagerank(self.inverted_graph, alpha=self.d, personalization=personalization, weight='p')
        # Initialize the value of delta with the influence-PageRank of the node
        delta_s = fPR[s]
        # For each node j in the seed set, subtract two contributions from the current value of delta:
        # 1) The influence-PageRank of node s multiplied by the weight of the edge (j,s), if exists
        # 2) The influence-PageRank of node j multiplied by the weight of the edge (s,j), if exists
        for j in seed_set:
            p_js = self.graph.edges[j, s]['p'] if self.graph.has_edge(j, s) else 0
            p_sj = self.graph.edges[s, j]['p'] if self.graph.has_edge(s, j) else 0
            delta_s = delta_s - self.d * p_js * fPR[s] - self.d * p_sj * fPR[j]
        # Formula inside the paper
        return delta_s * (len(self.influencee) / (1 - self.d))

    def run(self):
        """
        :return: The nodes to add in the seed set of the current agent and the spreads for each agent.
        """
        # This method is necessary since when the input network is signed, the graph of the proxy-based algorithm
        # contains only the trust-edges (see super-class ProxyBasedAlgorithm)
        self.__update_active_nodes__()
        # Initialize the delta dictionary if it's the first turn of the first round
        if self.delta_dict is None:
            self.__initialize_delta_dict__()
        agents_copy = copy.deepcopy(self.agents)
        # Repeats until the budget is fulfilled (in the InfluenceMaximization class, inside the run method, the algorithm
        # is always invoked with the budget parameter set to 1, but we preferred to write the code in a more general way)
        added_nodes = 0
        while added_nodes < self.budget:
            # Take the node which has the maximum value of delta. The heap property guarantees that
            # the first item is the one with the highest value of delta
            s, neg_delta = self.delta_dict[self.curr_agent_id].popitem()
            # Update this node's delta value with bound method and reinsert the node into the heap
            self.delta_dict[self.curr_agent_id][s] = -self.__get_delta_bound__(agents_copy[self.curr_agent_id].seed, s)
            # If it's still the node with the highest value of delta
            if s == self.delta_dict[self.curr_agent_id].peekitem()[0]:
                s_max, _ = self.delta_dict[self.curr_agent_id].peekitem()
                agents_copy[self.curr_agent_id].seed.append(s_max) # Add it into the seed set of the current agent
                self.__remove_node_from_heaps__(s_max) # And remove it from all the heaps
                self.influencee.remove(s_max) # Remove it also from the set of nodes that can be influenced
                added_nodes += 1
        # Return the new nodes to add to the seed set and the spread (which is 0 because we didn't do any simulation,
        # in fact this is only a fictional value, since the real spread will be computed at the end of the game)
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: 0 for a in self.agents}