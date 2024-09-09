from src.multi_agent.diffusion_models.diffusion_model import DiffusionModel
import src.multi_agent.competitive_influence_maximization as cim
import random

class IndependentCascade(DiffusionModel):
    """
    Paper: Goldenberg et al. - "Talk of the network: A complex system look at the underlying process of word-of-mouth"
    """

    name = 'ic'

    def __init__(self, endorsement_strategy):
        super().__init__(endorsement_strategy)

    def preprocess_data(self, graph):
        return

    def activate(self, graph, agents):
        """
        :return: A dictionary with the agents as keys and the list of nodes activated by each agent as values
        """
        sim_graph = graph.copy()
        for agent in agents:
            for u in agent.seed:
                cim.activate_node(sim_graph, u, agent)
        active_set = cim.active_nodes(sim_graph)
        newly_activated = list(active_set)
        while len(newly_activated) > 0:
            # First phase: try to influence inactive nodes
            # Each newly activated node tries to activate its inactive neighbors by contacting them
            for u in newly_activated:
                inactive_out_edges = [(u, v, attr) for (u, v, attr) in sim_graph.out_edges(u, data=True) if
                                      not cim.is_active(v, sim_graph)]
                for (_, v, attr) in inactive_out_edges:
                    r = random.random()
                    if r < attr['p']:
                        cim.contact_node(sim_graph, v, sim_graph.nodes[u]['agent'])
            # Second phase: contacted inactive nodes choose which agent to endorse by a strategy
            newly_activated = cim.manage_pending_nodes(sim_graph, self.endorsement_strategy)
            active_set.extend(newly_activated)
        # Group the activated nodes by agent and return the result
        return self.__group_by_agent__(sim_graph, active_set)