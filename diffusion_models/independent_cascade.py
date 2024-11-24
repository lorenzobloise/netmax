import random
from diffusion_models.diffusion_model import DiffusionModel
import influence_maximization as im

class IndependentCascade(DiffusionModel):
    """
    Paper: Goldenberg et al. - "Talk of the network: A complex system look at the underlying process of word-of-mouth"
    """

    name = 'ic'

    def __init__(self, endorsement_policy):
        super().__init__(endorsement_policy)

    def __copy__(self):
        result = IndependentCascade(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items(): # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def preprocess_data(self, graph):
        return

    def __activate_initial_nodes__(self, graph, agents):
        active_set = set()
        for agent in agents:
            for node in agent.seed:
                if not self.sim_graph.has_node(node):
                    self.__add_node__(graph, node)
                im.activate_node_in_simulation_graph(graph, self.sim_graph, node, agent)
                self.__add_node_to_the_stack__(node)
                active_set.add(node)
        return list(active_set)

    def activate(self, graph, agents):
        self.__reset_parameters__()
        if self.sim_graph is None:
            self.__initialize_sim_graph__(graph, agents)
        active_set = self.__activate_initial_nodes__(graph, agents)
        newly_activated = list(active_set)
        self.__register_history__(active_set, {})
        while len(newly_activated) > 0:
            # First phase: try to influence inactive nodes
            # Each newly activated node tries to activate its inactive neighbors by contacting them
            pending_nodes = []
            for u in newly_activated: #Consider the nodes activated at time step t-1.
                inactive_out_edges = self.__build_inactive_out_edges__(graph, u) #Get the inactive out-neighbors of u.
                for (_, v, attr) in inactive_out_edges: # For each inactive neighbor v of u
                    r = random.random()
                    if r < attr['p']: #For each inactive neighbor v of u, try to activate v, under the decreasing cascade model.
                        im.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        if v not in pending_nodes:
                            pending_nodes.append(v) # Add v to the pending nodes
            self.__extend_stack__(pending_nodes)
            self.__register_history__(None, pending_nodes)
            # Second phase: handle the pending nodes and designate the newly activated nodes as the ones activated in the current time step
            newly_activated = im.manage_pending_nodes(graph, self.sim_graph, self.endorsement_policy, pending_nodes)
            active_set.extend(newly_activated)
            self.__register_history__(active_set, {})
        result = self.__group_active_set_by_agent__(active_set)
        self.__reverse_operations__(graph)
        return result