from single_agent.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
from tqdm import tqdm
from single_agent import influence_maximization as im
import numpy as np

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