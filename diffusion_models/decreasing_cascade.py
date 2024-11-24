from diffusion_models.diffusion_model import DiffusionModel
import influence_maximization as im
import random

class DecreasingCascade(DiffusionModel):
    """
    Paper: Kempe et al. - "Influential Nodes in a Diffusion Model for Social Networks"
    """

    name = 'dc'

    def __init__(self, endorsement_policy):
        super().__init__(endorsement_policy)

    def __copy__(self):
        result = DecreasingCascade(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items(): # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def preprocess_data(self, graph):
        for node in graph.nodes:
            graph.nodes[node]['trials'] = 0

    def __initialize_sim_graph__(self, graph, agents):
        super().__initialize_sim_graph__(graph, agents)
        self.sim_graph.graph['stack_trials'] = set()  # Stack for dynamic probabilities

    def __activate__initial_nodes__(self, graph, agents):
        active_set = []
        for agent in agents:
            for u in agent.seed:
                if u not in self.sim_graph.nodes:
                    self.__add_node__(graph, u)
                im.activate_node_in_simulation_graph(graph, self.sim_graph, u, agent)
                active_set.append(u)
                self.__add_node_to_the_stack__(u)
                if 'trials' in self.sim_graph.nodes[u]:
                    del self.sim_graph.nodes[u]['trials']  # Remove the trials attribute of the node to avoid memory waste
                    self.sim_graph.graph['stack_trials'].add(u)
        return active_set

    def __reverse_operations__(self,graph):
        """
        This method empties the stack of the active nodes
        """
        super().__reverse_operations__(graph)
        stack_trials = self.sim_graph.graph['stack_trials']
        while len(stack_trials) > 0:
            node = stack_trials.pop()
            self.sim_graph.nodes[node]['trials'] = 0

    def activate(self, graph, agents):
        self.__reset_parameters__()
        if self.sim_graph is None:
            self.__initialize_sim_graph__(graph, agents)
        active_set = self.__activate__initial_nodes__(graph, agents)
        newly_activated = list(active_set)
        self.__register_history__(active_set, {})
        while len(newly_activated) > 0:
            # First phase: try to contact inactive nodes
            pending_nodes = []
            for u in newly_activated: #Consider the nodes activated at time step t-1.
                inactive_out_edges = self.__build_inactive_out_edges__(graph, u) #Get the inactive out-neighbors of u.
                for (_, v, attr) in inactive_out_edges: #For each inactive neighbor v of u, try to activate v, under the decreasing cascade model.
                    r = random.random()
                    trials = self.sim_graph.nodes[v]['trials']
                    if trials == 1:
                        self.sim_graph.graph['stack_trials'].add(v)
                    if r < attr['p'] * (1 / (0.1 * (trials ** 2) + 1)):
                        im.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        if v not in pending_nodes:
                            pending_nodes.append(v)
                    else:
                        self.sim_graph.nodes[v]['trials'] = trials + 1
            self.__extend_stack__(pending_nodes)
            self.__register_history__(None, pending_nodes)
            # Second phase: handle the pending nodes and designate the newly activated nodes as the ones activated in the current time step
            newly_activated = im.manage_pending_nodes(graph, self.sim_graph, self.endorsement_policy, pending_nodes)
            active_set.extend(newly_activated)
            self.__register_history__(active_set, {})
        result = self.__group_active_set_by_agent__(active_set)
        self.__reverse_operations__(graph)
        return result