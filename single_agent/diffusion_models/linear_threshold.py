from single_agent.diffusion_models.diffusion_model import DiffusionModel
import single_agent.influence_maximization as im
import random

class LinearThreshold(DiffusionModel):
    """
    Paper: Granovetter et al. - "Threshold models of collective behavior"
    """

    name = 'lt'

    def __init__(self):
        super().__init__()

    def preprocess_data(self, graph):
        for node in graph.nodes:
            graph.nodes[node]['threshold'] = random.random()
            graph.nodes[node]['prob_sum'] = 0

    def __update_prob_sum__(self, graph, node):
        del graph.nodes[node]['prob_sum'] # Remove the prob_sum dict of the node to avoid memory waste
        for (_, v, attr) in graph.out_edges(node, data=True):
            if not im.is_active(v, graph):
                graph.nodes[v]['prob_sum'] += attr['p']

    def activate(self, graph, agent, seed):
        """
        :return: A dictionary with the agents as keys and the list of nodes activated by each agent as values
        """
        sim_graph = graph.copy()
        active_set = []
        for u in seed:
            im.activate_node(sim_graph, u, agent)
            self.__update_prob_sum__(sim_graph, u)
            active_set.append(u)
        newly_activated = list(active_set)
        tmp = []
        while len(newly_activated) > 0:
            inactive_set = im.inactive_nodes(sim_graph)
            for u in inactive_set:
                if sim_graph.nodes[u]['prob_sum'] >= sim_graph.nodes[u]['threshold']:
                    im.activate_node(sim_graph, u, agent)
                    tmp.append(u)
            newly_activated = tmp
            active_set.extend(newly_activated)
            # Update the probability sum of the neighbour of newly activated nodes
            for u in newly_activated:
                self.__update_prob_sum__(sim_graph, u)
        return active_set