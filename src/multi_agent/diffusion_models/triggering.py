from diffusion_model import DiffusionModel
import src.multi_agent.competitive_influence_maximization as cim
import random

class Triggering(DiffusionModel):
    """
    Paper: Kempe et al. - "Maximizing the Spread of Influence through a Social Network"
    """

    name = 'tr'

    def __init__(self, endorsement_strategy):
        super().__init__(endorsement_strategy)

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

    def activate(self, graph, agents):
        """
        :return: A dictionary with the agents as keys and the list of nodes activated by each agent as values
        """
        sim_graph = graph.copy()
        for agent in agents:
            for u in agent.seed:
                cim.activate_node(sim_graph, u, agent.name)
        active_set = cim.active_nodes(sim_graph)
        newly_activated = list(active_set)
        while len(newly_activated) > 0:
            # First phase: try to influence inactive nodes
            # Each newly activated node tries to activate its inactive neighbors by contacting them
            for u in newly_activated:
                for v in sim_graph.nodes[u]['reverse_trigger_set']:
                    if not cim.is_active(v, sim_graph):
                        cim.contact_node(sim_graph, v, sim_graph.nodes[u]['agent'])
            # Second phase: contacted inactive nodes choose which agent to endorse by a strategy
            newly_activated = cim.manage_pending_nodes(sim_graph, self.endorsement_strategy)
            active_set.extend(newly_activated)
        # Group the activated nodes by agent and return the result
        return self.__group_by_agent__(sim_graph, active_set)