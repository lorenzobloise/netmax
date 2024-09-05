import numpy as np
from tqdm import tqdm
from src.single_agent.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm

class UBound(ProxyBasedAlgorithm):
    """
    Paper: Zhou et al. - "UBLF: An Upper Bound Based Approach to Discover Influential Nodes in Social Networks"
    The UBound algorithm doesn't work if the graph has cycles.
    """

    name = 'ubound'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def __get_propagation_probability_matrix(self, graph):
        """
        :param graph:
        :return: PP: propagation probability matrix
        """
        num_nodes = graph.number_of_nodes()
        PP = np.zeros((num_nodes, num_nodes))
        for (u, v, data) in graph.edges(data=True):
            PP[u][v] = data['p']
        return PP

    def __get_delta_appr__(self, graph):
        PP = self.__get_propagation_probability_matrix(graph)
        one_vector = np.ones((graph.number_of_nodes(), 1))
        a_t = [np.dot(PP, one_vector)]
        progress_bar = tqdm()
        while True:
            a_t.append(np.dot(PP, a_t[-1]))
            if np.linalg.norm(a_t[-1], ord=1) < 10 ** (-6):
                break
            progress_bar.update(1)
        progress_bar.close()
        return np.sum(a_t, axis=0)

    def __get_delta_exact__(self, graph):
        PP = self.__get_propagation_probability_matrix(graph)
        E = np.eye(PP.shape[0], PP.shape[1])
        delta_vector = np.dot(np.linalg.inv(E - PP), np.ones((graph.number_of_nodes(), 1)))
        return delta_vector

    def run(self):
        delta = self.__get_delta_exact__(self.graph)  # Vector N x 1
        delta_dict = {i: delta[i] for i in range(len(delta))}
        delta_dict = dict(sorted(delta_dict.items(), key=lambda item: item[1], reverse=True))
        seed_set = list(delta_dict.keys())[:self.budget]
        return seed_set