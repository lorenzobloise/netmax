import numpy as np
from tqdm import tqdm
from heapdict import heapdict
import networkx as nx
import influence_maximization as im
import math

class SimulationBasedAlgorithm:
    """
    Simulation-based algorithms for seed set selection in influence maximization problems rely on simulating the spread
    of influence through a network to identify the most influential nodes.
    These algorithms use Monte Carlo simulations to estimate the expected spread of influence for different sets of seed
    nodes. The goal is to select a set of nodes that maximizes the spread of influence within a given budget.
    """

    def __init__(self, graph, agent, budget, diff_model, r):
        self.graph = graph
        self.agent = agent
        self.budget = budget
        self.diff_model = diff_model
        self.r = r

    def set_graph(self, graph):
        self.graph = graph

    def set_agent(self, agent):
        self.agent = agent

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")


class MCGreedy(SimulationBasedAlgorithm):

    name = 'mcgreedy'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        sim_graph = self.graph.copy()
        seed = []
        spread = 0
        for _ in tqdm(range(self.budget), desc="Seed set"):
            marginal_gains = []
            for u in tqdm(im.inactive_nodes(sim_graph), desc="Inactive nodes"):
                tmp_seed = seed + [u]
                marginal_gain = im.simulation(sim_graph, self.diff_model, self.agent, tmp_seed, self.r) - spread
                marginal_gains.append((u,marginal_gain))
            marginal_gains = sorted(marginal_gains, key=lambda item: item[1], reverse=True)
            (top_node, top_marginal_gain) = marginal_gains.pop(0)
            seed.append(top_node)
            im.activate_node(sim_graph, top_node, self.agent)
            spread += top_marginal_gain
        return seed


class CELF(SimulationBasedAlgorithm):
    """
    Paper: Leskovec et al. - "Cost-Effective Outbreak Detection in Networks"
    """

    name = 'celf'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        sim_graph = self.graph.copy()
        seed = []
        marginal_gains = []
        # First iteration: Monte Carlo and store marginal gains
        for u in tqdm(im.inactive_nodes(sim_graph), desc="First Monte Carlo simulation"):
            tmp_seed = seed + [u]
            marginal_gains.append((u, im.simulation(sim_graph, self.diff_model, self.agent, tmp_seed, self.r)))
        marginal_gains = sorted(marginal_gains, key=lambda item: item[1], reverse=True)
        (top_node, spread) = marginal_gains.pop(0)
        seed.append(top_node)
        im.activate_node(sim_graph, top_node, self.agent)
        # Other iterations: use marginal gains to do an early stopping
        for _ in tqdm(range(self.budget-1), desc="Seed set"):
            check = False
            while not check:
                u = marginal_gains[0][0]
                tmp_seed = seed + [u]
                marginal_gain = im.simulation(sim_graph, self.diff_model, self.agent, tmp_seed, self.r) - spread
                marginal_gains[0] = (u, marginal_gain)
                marginal_gains = sorted(marginal_gains, key=lambda item: item[1], reverse=True)
                if marginal_gains[0][0] == u:
                    check = True
            (top_node, top_marginal_gain) = marginal_gains.pop(0)
            seed.append(top_node)
            spread += top_marginal_gain
        return seed


class CELF_PP(SimulationBasedAlgorithm):
    """
    Paper: Goyal et al. - "CELF++: Optimizing the Greedy Algorithm for Influence Maximization in Social Networks"
    """

    name = 'celfpp'

    class Node(object):

        def __init__(self, node):
            self.node = node
            self.mg1 = 0
            self.prev_best = None
            self.mg2 = 0
            self.mg2_already_computed = False
            self.flag = None
            self.list_index = 0

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        sim_graph = self.graph.copy()
        seed_set = []
        Q = heapdict() # Priority Queue based on marginal gain 1
        last_seed = None
        cur_best = None
        node_data_list = []
        # First iteration: Monte Carlo and store marginal gains
        for node in tqdm(sim_graph.nodes, desc="First Monte Carlo simulation"):
            node_data = CELF_PP.Node(node)
            node_data.mg1 = im.simulation(sim_graph, self.diff_model, self.agent, [node_data.node], self.r)
            node_data.prev_best = cur_best
            node_data.flag = 0
            cur_best = cur_best if cur_best and cur_best.mg1 > node_data.mg1 else node_data
            node_data_list.append(node_data)
            node_data.list_index = len(node_data_list) - 1
            Q[node_data.list_index] = -node_data.mg1
        # Other iterations: use marginal gains to do an early stopping
        progress_bar = tqdm(total=self.budget, desc="Seed Set", unit="it")
        while len(seed_set) < self.budget:
            node_idx, _ = Q.peekitem()
            node_data = node_data_list[node_idx]
            if not node_data.mg2_already_computed:
                node_data.mg2 = im.simulation(sim_graph, self.diff_model, self.agent, [node_data.node] + [cur_best.node], self.r)
                node_data.mg2_already_computed = True
            if node_data.flag == len(seed_set):
                seed_set.append(node_data.node)
                progress_bar.update(1)
                del Q[node_idx]
                last_seed = node_data
                continue
            elif node_data.prev_best == last_seed:
                node_data.mg1 = node_data.mg2
            else:
                node_data.mg1 = im.simulation_delta(sim_graph, self.diff_model, self.agent, seed_set+[node_data.node], seed_set, self.r)
                node_data.prev_best = cur_best
                node_data.mg2 = im.simulation_delta(sim_graph, self.diff_model, self.agent, seed_set+[cur_best.node]+[node_data.node], seed_set+[cur_best.node], self.r)
            node_data.flag = len(seed_set)
            cur_best = cur_best if cur_best and cur_best.mg1 > node_data.mg1 else node_data
            Q[node_idx] = -node_data.mg1
        return seed_set


class UBLF(SimulationBasedAlgorithm):
    """
    Paper: Zhou et al. - "UBLF: An Upper Bound Based Approach to Discover Influential Nodes in Social Networks"
    """

    name = 'ublf'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def __get_propagation_probability_matrix__(self, graph):
        """
        :param graph:
        :return: PP: propagation probability matrix
        """
        num_nodes = graph.number_of_nodes()
        PP = np.zeros((num_nodes,num_nodes))
        for (u, v, data) in graph.edges(data=True):
            PP[u][v] = data['p']
        return PP

    def __get_delta_appr__(self, graph):
        PP = self.__get_propagation_probability_matrix__(graph)
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
        PP = self.__get_propagation_probability_matrix__(graph)
        E = np.eye(PP.shape[0], PP.shape[1])
        delta = np.dot(np.linalg.inv(E - PP), np.ones((graph.number_of_nodes(), 1)))
        return delta

    def __get_max_nodes_delta_except_seed__(self, delta, seed_set):
        mask = np.ones(delta.shape, dtype=bool)
        mask[seed_set] = False
        delta_exclude = np.where(mask,delta,-np.inf)
        u = np.argmax(delta_exclude)
        return u

    def run(self):
        sim_graph = self.graph.copy()
        delta = self.get_delta_exact(sim_graph) # Vector N x 1
        seed_set = []
        for _ in tqdm(range(self.budget),desc="Seed set"):
            dict_I = {}
            for u in im.inactive_nodes(sim_graph):
                dict_I[u] = 0
            while True:
                # Take the node with the maximum value of delta not in the seed set
                u = self.__get_max_nodes_delta_except_seed__(delta, seed_set)
                if dict_I[u] == 0:
                    delta[u] = im.simulation_delta(sim_graph, self.diff_model, self.agent, seed_set+[u], seed_set, self.r)
                    dict_I[u] = 1
                if delta[u] >= delta[self.__get_max_nodes_delta_except_seed__(delta, seed_set+[u])]:
                    seed_set.append(u)
                    im.activate_node(sim_graph, u, self.agent)
                    break
        return seed_set

class CGA(SimulationBasedAlgorithm):
    """
    Paper: Wang et al. - "Community-based Greedy Algorithm for Mining Top-K Influential Nodes in Mobile Social Networks"
    """

    name = 'cga'

    def __init__(self, graph, agent, budget, diff_model, r=10000):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        sim_graph = self.graph.copy()
        seed_set = []
        # Detect communities
        C = nx.community.louvain_communities(sim_graph, weight='p')
        M = len(C)
        seed_communities = {i: [] for i in range(M)}
        R = {}
        s = {}
        for k in range(1,self.budget+1):
            R[(0,k)] = 0
            s[(0,k)] = 0
        for m in range(1,M+1):
            R[(m,0)] = 0
        for k in tqdm(range(1,self.budget+1), desc="Seed set"):
            for m in tqdm(range(1,M+1), desc="Communities"):
                spreads_m = [im.simulation_delta(sim_graph, self.diff_model, self.agent, seed_set+[v], seed_set, self.r, C[m-1]) for v in C[m-1]]
                # Take maximum marginal gain for the community m
                delta_R_m = max(spreads_m)
                R[(m,k)] = max(R[(m-1,k)],R[(M,k-1)]+delta_R_m)
                if R[(m-1,k)] >= R[(M,k-1)] + delta_R_m:
                    s[(m,k)] = s[(m-1,k)]
                else:
                    s[(m,k)] = m
            j = s[(M,k)]
            marg_gains = {v_i: im.simulation(sim_graph, self.diff_model, self.agent, seed_communities[j-1]+[v_i], self.r, C[j-1]) - im.simulation(sim_graph, self.diff_model, self.agent, seed_communities[j-1], self.r) for v_i in C[j-1]}
            v_max = max(marg_gains, key=marg_gains.get)
            seed_communities[j-1].append(v_max)
            seed_set.append(v_max)
        return seed_set

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
                # Choose all out nodes into topknodes
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