from diffusion_models import SemiProgressiveFriendFoeDynamicLinearThreshold
import influence_maximization as im

class NonProgressiveFriendFoeDynamicLinearThreshold(SemiProgressiveFriendFoeDynamicLinearThreshold):
    """
    Paper: Calio, Tagarelli - Complex influence propagation based on trust-aware dynamic linear threshold models
    """

    name = 'np_f2dlt'

    def __init__(self, endorsement_policy, biased=True):
        super().__init__(endorsement_policy, biased)
        self.T = 100

    def __copy__(self):
        result = NonProgressiveFriendFoeDynamicLinearThreshold(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items():  # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def __deactivate_node__(self, graph, node):
        agent_name = self.sim_graph.nodes[node]['agent'].name
        for (_, v) in self.trust_graph.out_edges(node):
            attr = graph.get_edge_data(node, v)
            influ_p = attr['p']
            self.sim_graph.nodes[v]['prob_sum_trusted'][agent_name] -= influ_p
        im.deactivate_node_in_simulation_graph(graph, self.sim_graph, node)

    def __check_deactivated_nodes__(self, graph, active_set, seed_sets, newly_activated):
        for node in active_set.difference(seed_sets):
            dict_prob_sum_trusted = self.sim_graph.nodes[node]['prob_sum_trusted']
            # If the prob_sum_trusted is less than the node's threshold for each campaign, the node switches to inactive
            should_be_deactivated = True
            for prob_sum_trusted in dict_prob_sum_trusted.values():
                if prob_sum_trusted >= self.sim_graph.nodes[node]['threshold']:
                    should_be_deactivated = False
                    break
            if should_be_deactivated:
                self.__deactivate_node__(graph, node)
            if node in newly_activated:
                newly_activated.remove(node)
        return newly_activated