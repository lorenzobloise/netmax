from netmax.diffusion_models.diffusion_model import DiffusionModel
from netmax import influence_maximization as im
import random

class Triggering(DiffusionModel):
    """
    Paper: Kempe et al. - "Maximizing the Spread of Influence through a Social Network"
    """

    name = 'tr'

    def __init__(self, endorsement_policy):
        super().__init__(endorsement_policy)

    def __copy__(self):
        """
        Deep copy of the diffusion model
        """
        result = Triggering(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items(): # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def preprocess_data(self, graph):
        """
        For each node v, create a trigger set and a reverse trigger set. The trigger set consists of the in-neighbors u
        sampled according to the probability of the edge (u,v), while the reverse trigger set initially is empty and is
        gets updated while creating v's out-neighbors' trigger sets.
        :param graph: the original graph.
        """
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
        """
        Activates on the simulation graph the nodes which are already in some seed set.
        :param graph: the original graph.
        :param agents: the 'agents' dictionary.
        """
        active_set = set()
        for agent in agents:
            for node in agent.seed:
                if not self.sim_graph.has_node(node): # If the simulation graph doesn't already have the node
                    self.__add_node__(graph, node) # Add the node to the simulation graph
                im.activate_node_in_simulation_graph(graph, self.sim_graph, node, agent) # Activate it
                self.__add_node_to_the_stack__(node) # Add the node to the stack, to allow the operation reversal
                active_set.add(node) # And append it to the active set
        return list(active_set)

    def activate(self, graph, agents):
        """
        Performs a single simulation according to the specific diffusion model.
        :param graph: the original graph.
        :param agents: the 'agents' dictionary.
        :return: a dictionary <agent: active_nodes> which contains the active set for each agent.
        """
        # A simulation could have been performed before on this same diffusion model, so we reset the parameters
        self.__reset_parameters__()
        if self.sim_graph is None:
            self.__initialize_sim_graph__(graph, agents) # Initialize the simulation graph if it's the first time
        active_set = self.__activate_initial_nodes__(graph, agents) # Activate the nodes which already are in some seed set
        newly_activated = list(active_set)
        self.__register_history__(active_set, {}) # Register the initial state in the history
        while len(newly_activated)>0:
            pending_nodes = set()
            for u in newly_activated: # Consider the nodes activated at time step t-1.
                # Take nodes that 'u' activates, so check the reverse trigger set of 'u'.
                # For each node 'v' that 'u' activates, check if 'v' is not already active and if 'v' is not already put
                # in the pending nodes.
                for v in self.sim_graph.nodes[u]['reverse_trigger_set']:
                    if not self.sim_graph.has_node(v):
                        self.__add_node__(graph, v)
                        edge_attr = graph.get_edge_data(u, v)
                        self.sim_graph.add_edge(u, v, **edge_attr)
                        # Put the node in PENDING state
                        im.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        pending_nodes.add(v)
                    elif not im.is_active(v, self.sim_graph):
                        if not self.sim_graph.has_edge(u, v):
                            edge_attr = graph.get_edge_data(u, v)
                            self.sim_graph.add_edge(u, v, **edge_attr)
                        # Put the node in PENDING state
                        im.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        pending_nodes.add(v)
            self.__extend_stack__(pending_nodes) # Add the newly activated nodes to the stack
            self.__register_history__(None, pending_nodes) # Register the current state in the history
            # Second phase: resolve the competition by managing the pending nodes with the given endorsement policy
            newly_activated = im.manage_pending_nodes(graph, self.sim_graph, self.endorsement_policy, list(pending_nodes))
            active_set.extend(newly_activated) # Extend the active set with the newly activated nodes
            self.__register_history__(active_set, {}) # Register the current state in the history
        result = self.__group_active_set_by_agent__(active_set) # Build the result
        self.__reverse_operations__(graph) # Undo all the operations which are on the stack
        return result