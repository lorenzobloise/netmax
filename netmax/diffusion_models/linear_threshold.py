from netmax.diffusion_models.diffusion_model import DiffusionModel
import random
from netmax import influence_maximization as im


class LinearThreshold(DiffusionModel):
    """
    Paper: Granovetter et al. - "Threshold models of collective behavior"
    """

    name = 'lt'

    def __init__(self, endorsement_policy):
        super().__init__(endorsement_policy)

    def __copy__(self):
        """
        Deep copy of the diffusion model
        """
        result = LinearThreshold(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items(): # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def preprocess_data(self, graph):
        """
        For each node, sample the threshold from a uniform distribution in [0,1] and initialize the probability sum
        for each agent as a dictionary.
        :param graph: the original graph.
        """
        for node in graph.nodes:
            graph.nodes[node]['threshold'] = random.random()
            graph.nodes[node]['prob_sum'] = dict()

    def __initialize_sim_graph__(self, graph, agents):
        """
        Linear Threshold also needs a stack storing each node which has changed its 'prob_sum' dictionary, when
        one of its in-neighbor has been activated.
        :param graph: the original graph.
        :param agents: the 'agents' dictionary.
        """
        super().__initialize_sim_graph__(graph, agents) # Call the superclass method
        self.sim_graph.graph['stack_prob_sum'] = set()

    def __update_prob_sum__(self, graph, node, agent_name):
        """
        Updates the probability sum for the input node's out-neighbors, because it has been activated
        in the current iteration by the input agent.
        :param graph: the original graph.
        :param node: the node whose out neighbors' probability sum has to be updated.
        :param agent_name: the agent who activated this node.
        """
        # Delete the prob_sum dict of the newly activated node to avoid memory waste
        if 'prob_sum' in self.sim_graph.nodes[node]:
            del self.sim_graph.nodes[node]['prob_sum']
            self.__add_node_to_the_stack_prob_sum__(node)
        for (_, v, attr) in graph.out_edges(node, data=True):
            # If the neighbor is not on the simulation graph
            if not self.sim_graph.has_node(v):
                nodes_attr = graph.nodes(data=True)[v]
                self.sim_graph.add_node(v, **nodes_attr) # Add the node onto the simulation graph
                self.sim_graph.add_edge(node, v, **attr) # And add the corresponding edge
                # Update the neighbor's probability sum corresponding to the input agent with the edge weight
                self.sim_graph.nodes[v]['prob_sum'][agent_name] = self.sim_graph.nodes[v]['prob_sum'].get(agent_name, 0) + attr['p']
                if len(self.sim_graph.nodes[v]['prob_sum']) == 1:
                    # Equals to 1 if is the first time that the node is reached by someone
                    self.__add_node_to_the_stack_prob_sum__(v)
            # If the simulation graph contains the neighbor, but it's not active, do the same
            elif not im.is_active(v, self.sim_graph):
                self.sim_graph.nodes[v]['prob_sum'][agent_name] = self.sim_graph.nodes[v]['prob_sum'].get(agent_name, 0) + attr['p']
                if not self.sim_graph.has_edge(node, v):
                    self.sim_graph.add_edge(node, v, **attr)
                if len(graph.nodes[v]['prob_sum']) == 1:
                    # Equals to 1 if is the first time that the node is reached by someone
                    self.__add_node_to_the_stack_prob_sum__(v)

    def __add_node_to_the_stack_prob_sum__(self, node):
        """
        Adds a node to the 'prob_sum' stack.
        :param node: the node to add to the stack.
        """
        self.sim_graph.graph['stack_prob_sum'].add(node)

    def __activate_nodes_in_seed_sets__(self, graph, agents):
        """
        Activate the nodes in the seed sets of the agents in the simulation graph.
        :param graph: the original graph.
        :param agents: the 'agents' dictionary.
        """
        for agent in agents:
            for u in agent.seed:
                if not self.sim_graph.has_node(u): # If the simulation graph doesn't have the node, add it
                    self.__add_node__(graph, u)
                # Activate this node in the simulation graph
                im.activate_node_in_simulation_graph(graph, self.sim_graph, u, agent)
                self.__add_node_to_the_stack__(u) # Add the node to the stack
                self.__update_prob_sum__(graph, u, agent.name) # And update the probability sums of the neighbors

    def __reverse_operations__(self, graph):
        """
        This method empties the stack of the active nodes (superclass method call) and does the same with the
        'prob_sum' stack.
        :param graph: the original graph.
        """
        # Reset the prob_sum of the nodes that have been activated
        super().__reverse_operations__(graph)
        stack_prob_sum = self.sim_graph.graph['stack_prob_sum']
        while stack_prob_sum:
            node = stack_prob_sum.pop()
            self.sim_graph.nodes[node]['prob_sum'] = dict()

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
        self.__activate_nodes_in_seed_sets__(graph, agents) # Activate the nodes which already are in some seed set
        active_set = im.active_nodes(self.sim_graph)
        newly_activated = list(active_set)
        self.__register_history__(active_set, {}) # Register the initial state in the history
        while len(newly_activated) > 0:
            # First phase: try to contact inactive nodes
            pending_nodes = []
            for u in newly_activated: # Consider the nodes activated at time step t-1.
                curr_agent_name = self.sim_graph.nodes[u]['agent'].name
                # Expand the simulation graph by getting the inactive out-neighbors of u
                inactive_out_edges = self.__build_inactive_out_edges__(graph, u)
                for _, v, attr in inactive_out_edges: # For each inactive neighbor v of u, check if the threshold is reached
                    if self.sim_graph.nodes[v]['prob_sum'][curr_agent_name] >= self.sim_graph.nodes[v]['threshold']:
                        im.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        # Put the node in PENDING state
                        if v not in pending_nodes:
                            pending_nodes.append(v)
            self.__extend_stack__(pending_nodes) # Add the newly activated nodes to the stack
            self.__register_history__(None, pending_nodes) # Register the current state in the history
            # Second phase: resolve the competition by managing the pending nodes with the given endorsement policy
            newly_activated = im.manage_pending_nodes(graph, self.sim_graph, self.endorsement_policy, pending_nodes)
            active_set.extend(newly_activated) # Extend the active set with the newly activated nodes
            self.__register_history__(active_set, {}) # Register the current state in the history
            for u in newly_activated: # Each newly activated node updates the prob_sum of its neighbors
                self.__update_prob_sum__(graph, u, self.sim_graph.nodes[u]['agent'].name)
        result = self.__group_active_set_by_agent__(active_set) # Build the result
        self.__reverse_operations__(graph) # Undo all the operations which are on the stack
        return result