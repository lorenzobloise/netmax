from single_agent.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
from single_agent import influence_maximization as im
import networkx as nx
import math

# TODO: check
class CGINA(SimulationBasedAlgorithm):
    """
    Paper: Jiaguo LV et al. - A New Community-based Algorithm for Influence Maximization in Social Network
    """

    name = 'cgina'

    def __init__(self, graph, agent, budget, diff_model, r=10000):
        super().__init__(graph, agent, budget, diff_model, r)

    def __get_edges_across_different_communities__(self, graph):
        result = []
        for (u, v, attr) in graph.edges(data=True):
            if graph.nodes[u]['community'] != graph.nodes[v]['community']:
                result.append((u, v, attr))
        return result

    def __get_p_of_edges_inside_community__(self, graph, community):
        community_graph = graph.subgraph(community)
        return [attr['p'] for (u, v, attr) in list(community_graph.edges(data=True))]

    def __community_contribution__(self, graph, community):
        maximal_n_of_edges = len(community) * (len(community) - 1)
        wd = sum(self.__get_p_of_edges_inside_community__(graph, community)) / maximal_n_of_edges if maximal_n_of_edges != 0 else 0
        return wd

    def __shapley__(self, graph):
        """
        Calculate the Shapley value
            :param graph: The subgraph representing a community C_i.
            :return: The Shapley value of the community C_i.
        """

        communities = nx.community.louvain_communities(graph, weight='p')
        return sum([self.__community_contribution__(graph, c)*len(graph.nodes) for c in communities])

    def run(self):
        l = 0.2 # Proportion of bridge nodes in key nodes, chosen empirically as shown in the paper
        sim_graph = self.graph.copy()
        seed_set = []
        # Detect communities
        communities = nx.community.louvain_communities(sim_graph, weight='p', resolution=0.3)
        if len(communities) < self.budget:
            raise ValueError(f"Budget {self.budget} is less than the number of communities found ({len(communities)}). Set a smaller value for the budget.")
        # Preprocessing
        for i in range(len(communities)):
            for j in list(communities[i]):
                sim_graph.nodes[j]['community'] = i
                sim_graph.nodes[j]['bweight'] = 0
        shapley_values = {i: self.__shapley__(self.graph.subgraph(list(communities[i]))) for i in range(len(communities))} # Shapley value for every community
        out_nodes = {i: 0 for i in range(len(communities))} # Number of out nodes for every community
        key_nodes = {} # Number of key nodes for every community
        ssv = sum(list(shapley_values.values())) # Sum of Shapley values for all communities
        for i in range(len(communities)):
            key_nodes[i] = math.floor(shapley_values[i]/ssv)
            print(key_nodes[i])
        eb = self.__get_edges_across_different_communities__(sim_graph)
        for (u, v, attr) in eb:
            out_nodes[sim_graph.nodes[u]['community']] += 1
        for (u, v, attr) in eb:
            sim_graph.nodes[u]['bweight'] += attr['p']*shapley_values[sim_graph.nodes[v]['community']]
        influential_nodes = {} # Number of influential nodes in each community
        for i in range(len(communities)):
            if out_nodes[i] >= key_nodes[i]*l:
                # Choose the top key_nodes[c]*l nodes with the maximum bweight into topknodes
                influential_nodes[i] = math.floor(key_nodes[i] - key_nodes[i]*l) # key_nodes[c]*l is the number of bridge nodes in the community c
            else:
                # Choose all out nodes into top k nodes
                influential_nodes[i] = key_nodes[i] - out_nodes[i]
        while len(seed_set) < self.budget:
            marginal_gains = {}
            for i in range(len(communities)):
                for j in range(influential_nodes[i]):
                    marginal_gains[j] = im.simulation_delta(sim_graph, self.diff_model, self.agent, seed_set+[j], seed_set)
            v_max = sorted(marginal_gains.items(), key=lambda item: item[1], reverse=True)[0][0]
            seed_set.append(v_max)
            communities.remove(communities[sim_graph.nodes[v_max]['community']])
        return seed_set