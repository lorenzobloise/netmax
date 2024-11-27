import random
from netmax.diffusion_models.diffusion_model import DiffusionModel
from netmax import influence_maximization as im


class IndependentCascade(DiffusionModel):
    """
    Paper: Goldenberg et al. - "Talk of the network: A complex system look at the underlying process of word-of-mouth"
    """

    name = 'ic'

    def __init__(self, endorsement_policy):
        super().__init__(endorsement_policy)

    def __copy__(self):
        """
        Deep copy of the diffusion model
        """
        result = IndependentCascade(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items(): # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def preprocess_data(self, graph):
        """
        Independent Cascade doesn't need any particular preprocessing.
        :param graph: the original graph.
        """
        return

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
                im.activate_node_in_simulation_graph(graph, self.sim_graph, node, agent) # And activate it
                self.__add_node_to_the_stack__(node) # Add the node to the stack, to allow the operation reversal
                active_set.add(node)
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
            self.__initialize_sim_graph__(graph, agents)  # Initialize the simulation graph if it's the first time
        active_set = self.__activate_initial_nodes__(graph, agents) # Activate the nodes which already are in some seed set
        newly_activated = list(active_set)
        self.__register_history__(active_set, {}) # Register the initial state in the history
        while len(newly_activated) > 0:
            # First phase: try to influence inactive nodes
            # Each newly activated node tries to activate its inactive neighbors by contacting them
            pending_nodes = []
            for u in newly_activated: # Consider the nodes activated at time step t-1.
                # Expand the simulation graph by getting the inactive out-neighbors of u
                inactive_out_edges = self.__build_inactive_out_edges__(graph, u)
                for (_, v, attr) in inactive_out_edges: # For each inactive neighbor v of u
                    r = random.random()
                    if r < attr['p']: # Successful experiment, so put the node in PENDING state
                        im.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        if v not in pending_nodes:
                            pending_nodes.append(v) # Add v to the pending nodes
            self.__extend_stack__(pending_nodes) # Add the newly activated nodes to the stack
            self.__register_history__(None, pending_nodes) # Register the current state in the history
            # Second phase: resolve the competition by managing the pending nodes with the given endorsement policy
            newly_activated = im.manage_pending_nodes(graph, self.sim_graph, self.endorsement_policy, pending_nodes)
            active_set.extend(newly_activated) # Extend the active set with the newly activated nodes
            self.__register_history__(active_set, {}) # Register the current state in the history
        result = self.__group_active_set_by_agent__(active_set) # Build the result
        self.__reverse_operations__(graph) # Undo all the operations which are on the stack
        return result