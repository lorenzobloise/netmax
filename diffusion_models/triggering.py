from diffusion_models.diffusion_model import DiffusionModel
import influence_maximization as im
import random

class Triggering(DiffusionModel):
    """
    Paper: Kempe et al. - "Maximizing the Spread of Influence through a Social Network"
    """

    name = 'tr'

    def __init__(self, endorsement_policy):
        super().__init__(endorsement_policy)

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

    def __activate_initial_nodes__(self, graph, agents):
        active_set = set()
        for agent in agents:
            for node in agent.seed:
                if not self.sim_graph.has_node(node):
                    self.__add_node__(graph, node)
                im.activate_node(self.sim_graph, node, agent)
                self.__add_node_to_the_stack__(node)
                active_set.add(node)
        return list(active_set)


    def activate(self, graph, agents):
        if self.sim_graph is None:
            self.__initialize_sim_graph__(graph, agents)
        active_set = self.__activate_initial_nodes__(graph, agents)
        newly_activated = list(active_set)
        while len(newly_activated)>0:
            pending_nodes = set()
            for u in newly_activated:
                for v in self.sim_graph.nodes[u]['reverse_trigger_set']:
                    if not self.sim_graph.has_node(v):
                        self.__add_node__(graph, v)
                        edge_attr = graph.get_edge_data(u, v)
                        self.sim_graph.add_edge(u, v, **edge_attr)
                        im.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        pending_nodes.add(v)
                    elif not im.is_active(v, self.sim_graph):
                        if not self.sim_graph.has_edge(u, v):
                            edge_attr = graph.get_edge_data(u, v)
                            self.sim_graph.add_edge(u, v, **edge_attr)
                        im.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        pending_nodes.add(v)
            self.__extend_stack__(pending_nodes)
            newly_activated = im.manage_pending_nodes(self.sim_graph, self.endorsement_policy, list(pending_nodes))
            active_set.extend(newly_activated)
        result = self.__group_by_agent__(self.sim_graph, active_set)
        self.__reverse_operations__()
        return result