from src.single_agent.diffusion_models.diffusion_model import DiffusionModel
import src.single_agent.influence_maximization as im
import random

class IndependentCascade(DiffusionModel):
    """
    Paper: Goldenberg et al. - "Talk of the network: A complex system look at the underlying process of word-of-mouth"
    """

    name = 'ic'

    def __init__(self):
        super().__init__()

    def preprocess_data(self, graph):
        return

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
        while len(newly_activated) > 0:
            tmp = []
            for u in newly_activated:
                inactive_out_edges = [(u, v, attr) for (u, v, attr) in sim_graph.out_edges(u, data=True) if
                                      not im.is_active(v, sim_graph)]
                for (_, v, attr) in inactive_out_edges:
                    r = random.random()
                    if r < attr['p']:
                        im.activate_node(sim_graph, v, agent)
                        tmp.append(v)
            newly_activated = tmp
            active_set.extend(newly_activated)
        return active_set