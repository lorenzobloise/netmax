from single_agent.diffusion_models.diffusion_model import DiffusionModel
import single_agent.influence_maximization as im
import random

class Triggering(DiffusionModel):
    """
    Paper: Kempe et al. - "Maximizing the Spread of Influence through a Social Network"
    """

    name = 'tr'

    def __init__(self):
        super().__init__()

    def preprocess_data(self, graph):
        for node in graph.nodes:
            graph.nodes[node]['trigger_set'] = [] # Nodes that 'node' is activated by
            graph.nodes[node]['reverse_trigger_set'] = [] # Nodes that 'node' activates
        for v in graph.nodes:
            in_edges = graph.in_edges(v, data=True)
            for (u, _, edge_attr) in in_edges:
                r = random.random()
                if r < edge_attr['p']:
                    graph.nodes[v]['trigger_set'].append(u)
                    graph.nodes[u]['reverse_trigger_set'].append(v)

    def activate(self, graph, agent, seed):
        """
        :return: A dictionary with the agents as keys and the list of nodes activated by each agent as values
        """
        sim_graph = graph.copy()
        active_set = []
        for u in seed:
            im.activate_node(sim_graph, u, agent)
            active_set.append(u)
        newly_activated = list(active_set)
        tmp = []
        while len(newly_activated) > 0:
            for u in newly_activated:
                for v in sim_graph.nodes[u]['reverse_trigger_set']:
                    if not im.is_active(v, sim_graph):
                        im.activate_node(sim_graph, v, agent)
                        tmp.append(v)
            newly_activated = tmp
            active_set.extend(newly_activated)
        return active_set