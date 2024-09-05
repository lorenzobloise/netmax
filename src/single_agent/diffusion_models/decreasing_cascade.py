from src.single_agent.diffusion_models.diffusion_model import DiffusionModel
import src.single_agent.influence_maximization as im
import random

class DecreasingCascade(DiffusionModel):
    """
    Paper: Kempe et al. - "Influential Nodes in a Diffusion Model for Social Networks"
    """

    name = 'dc'

    def __init__(self):
        super().__init__()

    def preprocess_data(self, graph):
        for node in graph.nodes:
            graph.nodes[node]['trials'] = 0

    def activate(self, graph, agent, seed):
        """
        :return: A dictionary with the agents as keys and the list of nodes activated by each agent as values
        """
        sim_graph = graph.copy()
        active_set = []
        for u in seed:
            im.activate_node(sim_graph, u, agent)
            active_set.append(u)
            del sim_graph.nodes[u]['trials'] # Remove the trials attribute of the node to avoid memory waste
        active_set = im.active_nodes(sim_graph)
        newly_activated = list(active_set)
        tmp = []
        while len(newly_activated) > 0:
            for u in newly_activated:
                inactive_out_edges = [(u, v, attr) for (u, v, attr) in sim_graph.out_edges(u, data=True) if
                                      not im.is_active(v, sim_graph)]
                for (_, v, attr) in inactive_out_edges:
                    r = random.random()
                    trials = sim_graph.nodes[u]['trials']
                    if r < attr['p'] * (1 / (0.1 * (trials ** 2) + 1)):
                        im.activate_node(sim_graph, v, agent)
                        tmp.append(v)
                    else:
                        sim_graph.nodes[u]['trials'] = trials + 1
            newly_activated = tmp
            for u in newly_activated:
                del sim_graph.nodes[u]['trials'] # Remove the trials attribute of the node to avoid memory waste
            active_set.extend(newly_activated)
        return active_set