from diffusion_models.diffusion_model import DiffusionModel
import random
import math
import influence_maximization as im
import networkx as nx
from tqdm import tqdm

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
        self.T = 10
        self.graph_with_trust = None
        self.graph_with_distrust = None

    def __copy__(self):
        result = SemiProgressiveFriendFoeDynamicLinearThreshold(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items():  # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def preprocess_data(self, graph):
        for node in graph.nodes:
            graph.nodes[node]['threshold'] = random.random()
            graph.nodes[node]['prob_sum_trusted'] = dict()
            graph.nodes[node]['quiescence_time'] = random.uniform(0,5)
            graph.nodes[node]['quiescence_value'] = None # To be calculated once the node enters quiescent state
            graph.nodes[node]['last_activation_time'] = 0

    def __initialize_sim_graph__(self, graph, agents):
        super().__initialize_sim_graph__(graph, agents)
        self.sim_graph.graph['stack_last_activation_time'] = set()
        self.sim_graph.graph['stack_prob_sum_trusted'] = set()
        self.sim_graph.graph['stack_quiescence_value'] = set()

    def __add_node_to_the_stack_prob_sum_trusted__(self, node):
        self.sim_graph.graph['stack_prob_sum_trusted'].add(node)

    def __update_prob_sum_trusted__(self, graph, node, agent_name):
        # prob_sum_trusted dict must not be deleted, because it has to be updated to allow R3 state-transition rule
        if 'prob_sum_trusted' in self.sim_graph.nodes[node]:
            self.__add_node_to_the_stack_prob_sum_trusted__(node)
        for (_, v, attr) in self.graph_with_trust.out_edges(node, data=True):
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
        # At this point, all the node's out-neighbors have already been added to the simulation graph
        # For each of its out-neighbors (active or not) we redistribute the prob_sum_trusted
        for (_, v, attr) in self.graph_with_trust.out_edges(node, data=True):
            self.sim_graph.nodes[v]['prob_sum_trusted'][old_agent.name] = self.sim_graph.nodes[v]['prob_sum_trusted'].get(old_agent.name, 0) - attr['p']
            self.sim_graph.nodes[v]['prob_sum_trusted'][new_agent.name] = self.sim_graph.nodes[v]['prob_sum_trusted'].get(new_agent.name, 0) + attr['p']

    def __activate_nodes_in_seed_sets__(self, graph, agents):
        """
        Activate the nodes in the seed sets of the agents in the simulation graph
        """
        for agent in agents:
            for u in agent.seed:
                if not self.sim_graph.has_node(u):
                    self.__add_node__(graph, u)
                im.activate_node(self.sim_graph, u, agent)
                self.sim_graph.nodes[u]['last_activation_time'] = self.current_time
                self.__add_node_to_the_stack__(u)
                self.__update_prob_sum_trusted__(graph, u, agent.name)

    def __reverse_operations__(self):
        # Reset the prob_sum of the nodes that have been activated
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

    def __distrusted_in_neighbors_same_campaign__(self, graph, node):
        result = []
        for u in self.graph_with_distrust.predecessors(node):
            if (self.sim_graph.has_node(u)
                and im.is_active(u, self.sim_graph)
                and self.sim_graph.nodes[node]['agent'].name == self.sim_graph.nodes[u]['agent'].name):
                result.append(u)
        return result

    def __trusts__(self, graph, u, v):
        """
        Returns True if node u trusts his in-neighbor v
        """
        if not graph.has_edge(v, u):
            raise ValueError(f"Graph does not have edge ({v},{u})")
        return graph.edges[v, u]['p'] > 0

    def __quiescence_function__(self, graph, node):
        weight_sum = 0
        for u in self.__distrusted_in_neighbors_same_campaign__(graph, node):
            weight_sum += math.fabs(graph.edges[u, node]['p'])
        return graph.nodes[node]['quiescence_time'] + math.pow(math.e, self._lambda * weight_sum)

    def __activation_threshold_function__(self, graph, node, time):
        theta_v = graph.nodes[node]['threshold']
        if self.biased:
            return theta_v + self.delta * min((1 - theta_v)/self.delta, self.current_time - self.sim_graph.nodes[node]['last_activation_time'])
        else:
            exp_term = math.pow(math.e, -self._delta * (time - self.sim_graph.nodes[node]['last_activation_time'] - 1))
            indicator_func = 1 if time - self.sim_graph.nodes[node]['last_activation_time'] == 1 else 0
            return theta_v + exp_term - theta_v * indicator_func

    def __time_expired__(self):
        return self.current_time > self.T

    def __no_more_activation_attempts__(self, newly_activated, quiescent_nodes):
        if len(newly_activated) == 0 and len(quiescent_nodes) == 0:
            return True
        return False

    def __compute_quiescence_values__(self, graph, quiescent_nodes):
        for node in quiescent_nodes:
            self.sim_graph.nodes[node]['quiescence_value'] = math.floor(self.__quiescence_function__(graph, node))

    def __quiescence_expired__(self, node):
        self.sim_graph.nodes[node]['quiescence_value'] -= 1
        return self.sim_graph.nodes[node]['quiescence_value'] <= 0

    def __check_quiescent_nodes__(self, quiescent_nodes):
        """
        Check if any quiescent nodes has expired their quiescence state
        """
        result = []
        i = len(quiescent_nodes) - 1
        while i > 0:
            q = quiescent_nodes[i]
            if self.__quiescence_expired__(q):
                im.activate_node(self.sim_graph, q, self.sim_graph.nodes[q]['agent'])
                self.sim_graph.nodes[q]['last_activation_time'] = self.current_time
                result.append(quiescent_nodes.pop(i))
            i -= 1
        return result

    def __check_change_campaign__(self, graph, node, agents):
        """
        Check if the node should change the agent
        and if so, change it and return True
        otherwise return False
        """
        dict_prob_sum_trusted = self.sim_graph.nodes[node]['prob_sum_trusted']
        max_agent_name = max(dict_prob_sum_trusted, key=dict_prob_sum_trusted.get)
        if max_agent_name != self.sim_graph.nodes[node]['agent'].name:
            # Change the agent of the node
            old_agent= self.sim_graph.nodes[node]['agent']
            new_agent = None
            for agent in agents:
                if agent.name == max_agent_name:
                    new_agent = agent
                    break
            self.sim_graph.nodes[node]['agent'] = new_agent
            self.sim_graph.nodes[node]['last_activation_time'] = self.current_time
            # Update the prob_sum_trusted dict
            self.__redistribute_prob_sum_trusted__(graph, node, old_agent, new_agent)
            return True
        return False

    def __build_trusted_inactive_out_edges__(self, graph, u):
        inactive_out_edges = []
        for (_, v, attr) in self.graph_with_trust.out_edges(u, data=True):
            if not self.sim_graph.has_node(v):
                # If not in the simulation graph, is not active
                # because the node has not been reached yet
                inactive_out_edges.append((u, v, attr))
                # add the node
                nodes_attr = graph.nodes(data=True)[v]
                self.sim_graph.add_node(v, **nodes_attr)
                self.sim_graph.add_edge(u, v, **attr)
            elif not im.is_active(v, self.sim_graph):
                if not self.sim_graph.has_edge(u, v):
                    self.sim_graph.add_edge(u, v, **attr)
                inactive_out_edges.append((u, v, attr))
        return inactive_out_edges

    def activate(self, graph, agents):
        self.current_time = 0
        if self.sim_graph is None:
            self.__initialize_sim_graph__(graph, agents)
            self.graph_with_trust = nx.DiGraph()
            self.graph_with_distrust = nx.DiGraph()
            # Delete all the edges with p <=0
            graph_nodes = graph.nodes(data=True)
            for u, v, attr in graph.edges(data=True):
                node_u = graph_nodes[u]
                node_v = graph_nodes[v]
                if attr['p'] > 0:
                    self.graph_with_trust.add_node(u, **node_u)
                    self.graph_with_trust.add_node(v, **node_v)
                    self.graph_with_trust.add_edge(u, v, **attr)
                else:
                    self.graph_with_distrust.add_node(u, **node_u)
                    self.graph_with_distrust.add_node(v, **node_v)
                    self.graph_with_distrust.add_edge(u, v, **attr)


        self.__activate_nodes_in_seed_sets__(graph, agents)
        active_set = im.active_nodes(self.sim_graph)
        seed_sets = active_set.copy()
        newly_activated = list(active_set)
        quiescent_nodes = []
        progress_bar = tqdm(total=self.T, desc="Time")
        while not (self.__no_more_activation_attempts__(newly_activated, quiescent_nodes) or self.__time_expired__()):
            pending_nodes = []
            # R1 state-transition rule
            for u in newly_activated:
                curr_agent_name = self.sim_graph.nodes[u]['agent'].name
                inactive_out_edges = self.__build_trusted_inactive_out_edges__(graph, u)
                for _, v, attr in inactive_out_edges:
                    if self.sim_graph.nodes[v]['prob_sum_trusted'][curr_agent_name] >= self.__activation_threshold_function__(graph, v, self.current_time):
                        im.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        if v not in pending_nodes:
                            pending_nodes.append(v)
            # Contacted inactive nodes choose which campaign actually determines their transition in the quiescent state
            self.__extend_stack__(pending_nodes)
            quiescent_nodes.extend(im.transition_nodes_into_quiescent_state(self.sim_graph, self.endorsement_policy, pending_nodes))
            self.__compute_quiescence_values__(graph, quiescent_nodes)
            self.__extend_quiescence_stack__(quiescent_nodes)
            # R2 state-transition rule
            newly_activated = self.__check_quiescent_nodes__(quiescent_nodes)
            active_set.extend(newly_activated)
            for u in newly_activated:
                self.__update_prob_sum_trusted__(graph, u, self.sim_graph.nodes[u]['agent'].name)
            # R3 state-transition rule
            for u in active_set:
                if u in seed_sets:
                    continue
                if self.__check_change_campaign__(graph, u, agents) and u not in newly_activated:
                    newly_activated.append(u)
            self.current_time += 1
            progress_bar.update(1)
        progress_bar.update(self.T - self.current_time)
        result = self.__group_by_agent__(self.sim_graph, active_set)
        self.__reverse_operations__()
        return result
