from netmax.diffusion_models.diffusion_model import DiffusionModel
from netmax import influence_maximization as im
import random

class DecreasingCascade(DiffusionModel):
    """
    Paper: Kempe et al. - "Influential Nodes in a Diffusion Model for Social Networks"
    """

    name = 'dc'

    def __init__(self, endorsement_policy):
        super().__init__(endorsement_policy)

    def __copy__(self):
        """
        Deep copy of the diffusion model
        """
        result = DecreasingCascade(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items(): # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def preprocess_data(self, graph):
        """
        For each node, create an attribute 'trials', initialized to 0, that represents the number of times
        one of its neighbors has tried to influence the node.
        :param graph: the original graph.
        """
        for node in graph.nodes:
            graph.nodes[node]['trials'] = 0

    def __initialize_sim_graph__(self, graph, agents):
        """
        Decreasing Cascade also needs a stack storing the nodes who have changed the value of their 'trials' attribute.
        """
        super().__initialize_sim_graph__(graph, agents)
        self.sim_graph.graph['stack_trials'] = set()  # Stack for dynamic probabilities

    def __activate_initial_nodes__(self, graph, agents):
        """
        Activates on the simulation graph the nodes which are already in some seed set.
        :param graph: the original graph.
        :param agents: the 'agents' dictionary.
        """
        active_set = []
        for agent in agents:
            for u in agent.seed:
                if u not in self.sim_graph.nodes: # If the simulation graph doesn't already have the node
                    self.__add_node__(graph, u) # Add the node to the simulation graph
                im.activate_node_in_simulation_graph(graph, self.sim_graph, u, agent) # Activate it
                active_set.append(u) # And append it to the active set
                self.__add_node_to_the_stack__(u) # Add the node to the stack, to allow the operation reversal
                if 'trials' in self.sim_graph.nodes[u]:
                    del self.sim_graph.nodes[u]['trials']  # Remove the trials attribute of the node to avoid memory waste
                    self.sim_graph.graph['stack_trials'].add(u)
        return active_set

    def __reverse_operations__(self,graph):
        """
        This method empties the stack of the active nodes (superclass method call) and does the same with the
        'stack_trials' stack.
        :param graph: the original graph.
        """
        # Reset the trials of the nodes that have been activated
        super().__reverse_operations__(graph)
        stack_trials = self.sim_graph.graph['stack_trials']
        while len(stack_trials) > 0:
            node = stack_trials.pop()
            self.sim_graph.nodes[node]['trials'] = 0

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
        while len(newly_activated) > 0:
            # First phase: try to contact inactive nodes
            pending_nodes = []
            for u in newly_activated: # Consider the nodes activated at time step t-1.
                # Expand the simulation graph by getting the inactive out-neighbors of u
                inactive_out_edges = self.__build_inactive_out_edges__(graph, u)
                for (_, v, attr) in inactive_out_edges: # For each inactive neighbor v of u, try to activate v under the DC model
                    r = random.random()
                    trials = self.sim_graph.nodes[v]['trials']
                    if trials == 1:
                        self.sim_graph.graph['stack_trials'].add(v)
                    if r < attr['p'] * (1 / (0.1 * (trials ** 2) + 1)):
                        im.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        # Put the node in PENDING state
                        if v not in pending_nodes:
                            pending_nodes.append(v)
                    else:
                        self.sim_graph.nodes[v]['trials'] = trials + 1
            self.__extend_stack__(pending_nodes) # Add the newly activated nodes to the stack
            self.__register_history__(None, pending_nodes) # Register the current state in the history
            # Second phase: resolve the competition by managing the pending nodes with the given endorsement policy
            newly_activated = im.manage_pending_nodes(graph, self.sim_graph, self.endorsement_policy, pending_nodes)
            active_set.extend(newly_activated) # Extend the active set with the newly activated nodes
            self.__register_history__(active_set, {}) # Register the current state in the history
        result = self.__group_active_set_by_agent__(active_set) # Build the result
        self.__reverse_operations__(graph) # Undo all the operations which are on the stack
        return result