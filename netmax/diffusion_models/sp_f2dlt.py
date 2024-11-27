from netmax.diffusion_models.diffusion_model import DiffusionModel
import random
import math
from netmax import influence_maximization as im


class SemiProgressiveFriendFoeDynamicLinearThreshold(DiffusionModel):
    """
    Paper: Calio, Tagarelli - Complex influence propagation based on trust-aware dynamic linear threshold models
    """

    name = 'sp_f2dlt'

    def __init__(self, endorsement_policy, biased=False):
        super().__init__(endorsement_policy)
        self.biased = biased
        if self.biased:
            self.delta = 0.1 # Confirmation bias
        else:
            self._delta = 1 # Unbiased scenario
        self._lambda = random.uniform(0,5)
        self.current_time = 0
        self.T = 4
        self.trust_graph = None
        self.distrust_graph = None
        self.last_quiescent_set = None

    def __copy__(self):
        """
        Deep copy of the diffusion model
        """
        result = SemiProgressiveFriendFoeDynamicLinearThreshold(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items(): # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def preprocess_data(self, graph):
        """
        For each node, sample the threshold from a uniform distribution in [0,1], and initialize the probability sum
        for each agent as a dictionary (only consisting of trusted edges), the quiescence time, the quiescence
        value and the last activation time.
        :param graph: the original graph.
        """
        for node in graph.nodes:
            graph.nodes[node]['threshold'] = random.random()
            graph.nodes[node]['prob_sum_trusted'] = dict()
            graph.nodes[node]['quiescence_time'] = random.uniform(0,5)
            graph.nodes[node]['quiescence_value'] = None # To be calculated once the node enters quiescent state
            graph.nodes[node]['last_activation_time'] = 0

    def __initialize_sim_graph__(self, graph, agents):
        """
        SP-F2DLT also needs a stack storing each node which has changed their 'prob_sum_trusted', 'last_activation_time'
        and 'quiescence_value' dictionaries.
        :param graph: the original graph.
        :param agents: the 'agents' dictionary.
        """
        super().__initialize_sim_graph__(graph, agents) # Call the superclass method
        self.sim_graph.graph['stack_last_activation_time'] = set()
        self.sim_graph.graph['stack_prob_sum_trusted'] = set()
        self.sim_graph.graph['stack_quiescence_value'] = set()

    def __add_node_to_the_stack_prob_sum_trusted__(self, node):
        self.sim_graph.graph['stack_prob_sum_trusted'].add(node)

    def __update_prob_sum_trusted__(self, graph, node, agent_name):
        """
        Updates the probability sum for the input node's trusted out-neighbors, because it has been activated
        in the current iteration by the input agent.
        :param graph: the original graph.
        :param node: the node whose out neighbors' probability sum has to be updated.
        :param agent_name: the agent who activated this node.
        """
        # The prob_sum_trusted dict must not be deleted, because it has to be updated to allow R3 state-transition rule
        if 'prob_sum_trusted' in self.sim_graph.nodes[node]:
            self.__add_node_to_the_stack_prob_sum_trusted__(node)
        for (_, v) in self.trust_graph.out_edges(node):
            attr = graph.get_edge_data(node, v)
            # If v has not been added to the simulation graph yet, add it
            if not self.sim_graph.has_node(v):
                nodes_attr = graph.nodes[v]
                self.sim_graph.add_node(v, **nodes_attr)
                self.sim_graph.add_edge(node, v, **attr)
            if not self.sim_graph.has_edge(node, v):
                self.sim_graph.add_edge(node, v, **attr)
            # Update prob_sum_trusted dict
            self.sim_graph.nodes[v]['prob_sum_trusted'][agent_name] = self.sim_graph.nodes[v]['prob_sum_trusted'].get(agent_name, 0) + attr['p']
            if len(graph.nodes[v]['prob_sum_trusted']) == 1:
                # Equals to 1 if is the first time that the node is reached by someone
                self.__add_node_to_the_stack_prob_sum_trusted__(v)
                
    def __redistribute_prob_sum_trusted__(self, graph, node, old_agent, new_agent):
        """
        At this point, all the node's out-neighbors have already been added to the simulation graph, so for each of its
        out-neighbors (active or not) we redistribute the prob_sum_trusted.
        :param graph: the original graph.
        :param node: the node whose out neighbors' probability sum has to be redistributed.
        :param old_agent: the agent who previously activated this node.
        :param new_agent: the agent who now activated this node.
        """
        for (_, v) in self.trust_graph.out_edges(node):
            attr = graph.get_edge_data(node, v)
            self.sim_graph.nodes[v]['prob_sum_trusted'][old_agent.name] = self.sim_graph.nodes[v]['prob_sum_trusted'].get(old_agent.name, 0) - attr['p']
        im.deactivate_node_in_simulation_graph(graph, self.sim_graph, node)
        im.activate_node_in_simulation_graph(graph, self.sim_graph, node, new_agent)
        for (_, v) in self.trust_graph.out_edges(node):
            attr = graph.get_edge_data(node, v)
            self.sim_graph.nodes[v]['prob_sum_trusted'][new_agent.name] = self.sim_graph.nodes[v]['prob_sum_trusted'].get(new_agent.name, 0) + attr['p']

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
                self.sim_graph.nodes[u]['last_activation_time'] = self.current_time
                self.__add_node_to_the_stack__(u) # Add the node to the stack
                self.__update_prob_sum_trusted__(graph, u, agent.name) # And update the probability sums of the neighbors

    def __reverse_operations__(self, graph):
        """
        This method empties the stack of the active nodes (superclass method call) and does the same with the
        'prob_sum' stack.
        :param graph: the original graph.
        """
        # Reset the prob_sum and the other attributes of the nodes that have been activated
        stack_active_nodes = self.sim_graph.graph['stack_active_nodes']
        while len(stack_active_nodes) > 0:
            node = stack_active_nodes.pop()
            im.deactivate_node(self.sim_graph, node)
            self.sim_graph.nodes[node]['last_activation_time'] = 0
        stack_prob_sum = self.sim_graph.graph['stack_prob_sum_trusted']
        while stack_prob_sum:
            node = stack_prob_sum.pop()
            self.sim_graph.nodes[node]['prob_sum_trusted'] = dict()
        stack_quiescence_value = self.sim_graph.graph['stack_quiescence_value']
        while stack_quiescence_value:
            node = stack_quiescence_value.pop()
            self.sim_graph.nodes[node]['quiescence_value'] = None

    def __extend_quiescence_stack__(self, quiescent_nodes):
        self.sim_graph.graph['stack_quiescence_value'].update(quiescent_nodes)

    def __distrusted_in_neighbors_same_campaign__(self, node):
        """
        Returns the nodes that the input node doesn't trust, but that are in its same campaign.
        :param node: the input node.
        """
        result = []
        if not self.distrust_graph.has_node(node):
            # So the node is not in the distrust graph and has no distrust in-neighbors
            return result
        for u in self.distrust_graph.predecessors(node):
            if (self.sim_graph.has_node(u)
                and im.is_active(u, self.sim_graph)
                and self.sim_graph.nodes[node]['agent'].name == self.sim_graph.nodes[u]['agent'].name):
                result.append(u)
        return result

    def __quiescence_function__(self, graph, node):
        """
        This method defines the quiescence function as stated in the paper, with a penalization coming from the
        distrusted in neighbors who are in the same campaign as the input node.
        :param graph: the original graph.
        :param node: the input node.
        """
        weight_sum = 0
        for u in self.__distrusted_in_neighbors_same_campaign__(node):
            weight_sum += math.fabs(self.distrust_graph.edges[u, node]['p'])
        return graph.nodes[node]['quiescence_time'] + math.exp(self._lambda * weight_sum)

    def __activation_threshold_function__(self, node, time):
        """
        The value of this function determines whether a node will change campaign or not.
        :param node: the input node.
        :param time: the current time.
        """
        theta_v = self.sim_graph.nodes[node]['threshold']
        # Two possible scenarios:
        # 1) Confirmation bias
        if self.biased:
            return theta_v + self.delta * min((1 - theta_v)/self.delta, self.current_time - self.sim_graph.nodes[node]['last_activation_time'])
        # 2) Not biased
        else:
            exp_term = math.exp(-self._delta * (time - self.sim_graph.nodes[node]['last_activation_time'] - 1))
            indicator_func = 1 if time - self.sim_graph.nodes[node]['last_activation_time'] == 1 else 0
            return theta_v + exp_term - theta_v * indicator_func

    def __time_expired__(self):
        return self.current_time > self.T

    def __no_more_activation_attempts__(self, newly_activated, quiescent_nodes):
        """
        Returns True if there are no more activation attempts.
        :param newly_activated: the nodes who have just been activated.
        :param quiescent_nodes: the nodes who are in the QUIESCENT state.
        """
        if len(newly_activated) == 0 and len(quiescent_nodes) == 0:
            return True
        return False

    def __compute_quiescence_values__(self, graph, quiescent_nodes):
        """
        Computes the quiescence values for all the quiescent nodes.
        :param graph: the original graph.
        :param quiescent_nodes: the nodes who are in QUIESCENT state.
        """
        for node in quiescent_nodes:
            self.sim_graph.nodes[node]['quiescence_value'] = math.floor(self.__quiescence_function__(graph, node))

    def __quiescence_expired__(self, node):
        """
        Decrement the quiescence value of the input node and checks if it can exit the QUIESCENT state.
        :param node: the input node.
        """
        self.sim_graph.nodes[node]['quiescence_value'] -= 1
        return self.sim_graph.nodes[node]['quiescence_value'] <= 0

    def __check_quiescent_nodes__(self, graph, quiescent_nodes):
        """
        Check if any quiescent node has expired their quiescence state.
        :param graph: the original graph.
        :param quiescent_nodes: the nodes who are in QUIESCENT state.
        """
        newly_activated = set()
        # Iterating from the end of the list, to allow deleting the element safely.
        i = len(quiescent_nodes) - 1
        while i > 0:
            q = quiescent_nodes[i]
            if self.__quiescence_expired__(q):
                im.activate_node_in_simulation_graph(graph, self.sim_graph, q, self.sim_graph.nodes[q]['agent'])
                self.sim_graph.nodes[q]['last_activation_time'] = self.current_time
                newly_activated.add(quiescent_nodes.pop(i))
            i -= 1
        return newly_activated

    def __check_change_campaign__(self, graph, node, agents):
        """
        Check if the input node should change the agent and, if so, change it and return True,
        otherwise return False.
        :param graph: the original graph.
        :param node: the input node.
        :param agents: the 'agents' dictionary.
        """
        dict_prob_sum_trusted = self.sim_graph.nodes[node]['prob_sum_trusted']
        max_agent_name = max(dict_prob_sum_trusted, key=dict_prob_sum_trusted.get)
        if max_agent_name != self.sim_graph.nodes[node]['agent'].name and dict_prob_sum_trusted[max_agent_name] >= self.__activation_threshold_function__(node, self.current_time):
            # Change the agent of the node
            old_agent= self.sim_graph.nodes[node]['agent']
            new_agent = None
            for agent in agents:
                if agent.name == max_agent_name:
                    new_agent = agent
                    break
            self.sim_graph.nodes[node]['last_activation_time'] = self.current_time
            # Update the prob_sum_trusted dict
            self.__redistribute_prob_sum_trusted__(graph, node, old_agent, new_agent)
            return True
        return False

    def __build_trusted_inactive_out_edges__(self, graph, u):
        """
        Builds a list of trusted out edges, each one of these links the input node with another node which has not been
        activated yet.
        :param graph: the original graph.
        :param u: the input node.
        :return: the list of out edges linked to inactive nodes.
        """
        inactive_out_edges = []
        for (_, v) in self.trust_graph.out_edges(u):
            attr = graph.get_edge_data(u, v)
            # If the neighbor is not in the simulation graph, it isn't active because it hasn't been reached yet
            if not self.sim_graph.has_node(v):
                inactive_out_edges.append((u, v, attr))
                # So add the neighbor and the corresponding edge to the simulation graph
                nodes_attr = graph.nodes(data=True)[v]
                self.sim_graph.add_node(v, **nodes_attr)
                self.sim_graph.add_edge(u, v, **attr)
            # If the neighbor is in the simulation graph, but isn't active, we add the corresponding edge to the list
            elif not im.is_active(v, self.sim_graph) and not im.is_quiescent(v, self.sim_graph):
                if not self.sim_graph.has_edge(u, v): # If the simulation graph does not have the edge yet
                    self.sim_graph.add_edge(u, v, **attr) # Add the edge
                inactive_out_edges.append((u, v, attr))
        return inactive_out_edges

    def __check_deactivated_nodes__(self, graph, active_set, seed_sets, newly_activated):
        """
        This model does not have a state-transition rule that deactivates nodes (Semi-Progressive), so just return the
        newly activated nodes.
        :param graph: the original graph.
        :param active_set: the set of active nodes.
        :param seed_sets: the seed sets of the agents.
        :param newly_activated: the set of nodes that have just been activated.
        """
        return newly_activated

    def __register_history__(self, active_set, pending_set):
        raise Exception("This method should not be called in this diffusion model, use __register_history_with_quiescent_nodes__ instead")

    def __register_history_with_quiescent_nodes__(self, active_set, pending_set, quiescent_set):
        """
        Custom method definition for the diffusion models with the QUIESCENT state: in the history we also store
        the moment when nodes enter the QUIESCENT state.
        :param active_set: the set of active nodes.
        :param pending_set: the set of pending nodes.
        :param quiescent_set: the set of quiescent nodes.
        """
        if active_set is not None:
            self.last_active_set=self.__group_active_set_by_agent__(active_set)
        if quiescent_set is not None:
            self.last_quiescent_set = self.__group_quiescent_set_by_agent__(quiescent_set)
        self.history[self.iteration_id] = (self.last_active_set,  self.__build_pending_set_for_history__(pending_set), self.last_quiescent_set)
        self.iteration_id += 1

    def __group_quiescent_set_by_agent__(self, quiescent_set):
        """
        Groups the set of quiescent nodes by agent and returns it.
        :param quiescent_set: the set of quiescent nodes.
        """
        dict_result = {}
        for u in quiescent_set:
            curr_agent = self.sim_graph.nodes[u]['agent'].name
            if curr_agent in dict_result:
                dict_result[curr_agent].append(u)
            else:
                dict_result[curr_agent] = [u]
        return dict_result

    def activate(self, graph, agents):
        """
        Performs a single simulation according to the specific diffusion model.
        :param graph: the original graph.
        :param agents: the 'agents' dictionary.
        :return: a dictionary <agent: active_nodes> which contains the active set for each agent.
        """
        # A simulation could have been performed before on this same diffusion model, so we reset the parameters
        self.__reset_parameters__()
        self.current_time = 0
        if self.sim_graph is None:
            self.__initialize_sim_graph__(graph, agents) # Initialize the simulation graph if it's the first time
            # Build the trust and distrust graphs for better computation efficiency
            self.trust_graph, self.distrust_graph = im.build_trust_and_distrust_graphs(graph, verbose=False)
        self.__activate_nodes_in_seed_sets__(graph, agents) # Activate the nodes which already are in some seed set
        active_set = set(im.active_nodes(self.sim_graph))
        seed_sets = set(active_set.copy())
        newly_activated = set(active_set.copy())
        quiescent_nodes = []
        # Register the initial state in the history
        self.__register_history_with_quiescent_nodes__(active_set, {}, {})
        while not (self.__no_more_activation_attempts__(newly_activated, quiescent_nodes) or self.__time_expired__()):
            pending_nodes = set()
            # R1 state-transition rule
            for u in newly_activated:
                curr_agent_name = self.sim_graph.nodes[u]['agent'].name
                inactive_out_edges = self.__build_trusted_inactive_out_edges__(graph, u)
                for _, v, attr in inactive_out_edges:
                    if self.sim_graph.nodes[v]['prob_sum_trusted'][curr_agent_name] >= self.__activation_threshold_function__(v, self.current_time):
                        im.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        pending_nodes.add(v)
            # Contacted inactive nodes choose which campaign actually determines their transition in the quiescent state
            self.__extend_stack__(pending_nodes)
            self.__register_history_with_quiescent_nodes__(active_set, pending_nodes, None)
            quiescent_nodes.extend(im.transition_nodes_into_quiescent_state(self.sim_graph, self.endorsement_policy, pending_nodes))
            self.__compute_quiescence_values__(graph, quiescent_nodes)
            self.__extend_quiescence_stack__(quiescent_nodes)
            self.__register_history_with_quiescent_nodes__(None, {}, quiescent_nodes)
            # R2 state-transition rule
            newly_activated = self.__check_quiescent_nodes__(graph, quiescent_nodes)
            active_set.update(newly_activated)
            self.__register_history_with_quiescent_nodes__(active_set, {}, quiescent_nodes)
            for u in newly_activated:
                self.__update_prob_sum_trusted__(graph, u, self.sim_graph.nodes[u]['agent'].name)
            # R3 state-transition rule
            for u in active_set:
                if u in seed_sets:
                    continue
                if u not in newly_activated and self.__check_change_campaign__(graph, u, agents):
                    newly_activated.add(u)
            # R4 state-transition rule (implemented in subclass npF2DLT)
            newly_activated = self.__check_deactivated_nodes__(graph, active_set, seed_sets, newly_activated)
            self.current_time += 1
        result = self.__group_active_set_by_agent__(active_set)
        self.__reverse_operations__(graph)
        return result
