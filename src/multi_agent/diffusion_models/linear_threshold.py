from src.multi_agent.diffusion_models.diffusion_model import DiffusionModel
import src.multi_agent.competitive_influence_maximization as cim
import random

class LinearThreshold(DiffusionModel):
    """
    Paper: Granovetter et al. - "Threshold models of collective behavior"
    """

    name = 'lt'

    def __init__(self, endorsement_strategy):
        super().__init__(endorsement_strategy)

    def preprocess_data(self, graph):
        for node in graph.nodes:
            graph.nodes[node]['threshold'] = random.random()
            graph.nodes[node]['prob_sum'] = dict()

    def __update_prob_sum__(self, graph, node, agent_name):
        del graph.nodes[node]['prob_sum'] # Remove the prob_sum dict of the node to avoid memory waste
        for (_, v, attr) in graph.out_edges(node, data=True):
            if not cim.is_active(v, graph):
                if agent_name not in graph.nodes[v]['prob_sum']:
                    graph.nodes[v]['prob_sum'][agent_name] = 0
                graph.nodes[v]['prob_sum'][agent_name] += attr['p']

    def activate(self, graph, agents):
        """
        :return: A dictionary with the agents as keys and the list of nodes activated by each agent as values
        """
        sim_graph = graph.copy()
        for agent in agents:
            for u in agent.seed:
                cim.activate_node(sim_graph, u, agent.name)
                self.__update_prob_sum__(sim_graph, u, agent.name)
        active_set = cim.active_nodes(sim_graph)
        newly_activated = list(active_set)
        while len(newly_activated) > 0:
            inactive_set = cim.inactive_nodes(sim_graph)
            # First phase: try to influence inactive nodes
            # Each newly activated node tries to activate its inactive neighbors by contacting them
            for u in inactive_set:
                curr_agent_name = sim_graph.nodes[u]['agent']
                if sim_graph.nodes[u]['prob_sum'][curr_agent_name] >= sim_graph.nodes[u]['threshold']:
                    cim.contact_node(sim_graph, u, curr_agent_name)
            # Second phase: contacted inactive nodes choose which agent to endorse by a strategy
            newly_activated = cim.manage_pending_nodes(sim_graph, self.endorsement_strategy)
            active_set.extend(newly_activated)
            # Update the probability sum of the neighbour of newly activated nodes
            for u in newly_activated:
                self.__update_prob_sum__(sim_graph, u, sim_graph.nodes[u]['agent'])
        # Group the activated nodes by agent and return the result
        return self.__group_by_agent__(sim_graph, active_set)