from diffusion_models.diffusion_model import DiffusionModel
import random
import competitive_influence_maximization as cim

class LinearThreshold(DiffusionModel):
    """
    Paper: Granovetter et al. - "Threshold models of collective behavior"
    """

    name = 'lt'

    def __init__(self, endorsement_policy):
        super().__init__(endorsement_policy)

    def __copy__(self):
        result = LinearThreshold(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items():  # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def preprocess_data(self, graph):
        for node in graph.nodes:
            graph.nodes[node]['threshold'] = random.random()
            graph.nodes[node]['prob_sum'] = dict()

    def __initialize_sim_graph__(self, graph, agents):
        super().__initialize_sim_graph__(graph, agents)
        self.sim_graph.graph['stack_prob_sum'] = set()


    def __update_prob_sum__(self,graph, node, agent_name):
        #delete the prob_sum dict of the newly activated node to avoid memory waste
        if 'prob_sum' in self.sim_graph.nodes[node]:
            del self.sim_graph.nodes[node]['prob_sum']
            self.__add_node_to_the_stack_prob_sum__(node)

        # I need to remember that i need to reset the prob_sum at the end of the simulation
        #for each neighbor of the newly activated node
        for (_, v, attr) in graph.out_edges(node, data=True):
            #if the neighbor is not active
            if not self.sim_graph.has_node(v):
                nodes_attr = graph.nodes(data=True)[v]
                self.sim_graph.add_node(v, **nodes_attr)
                self.sim_graph.add_edge(node, v, **attr)
                self.sim_graph.nodes[v]['prob_sum'][agent_name] = self.sim_graph.nodes[v]['prob_sum'].get(agent_name, 0) + attr['p']
                if len(self.sim_graph.nodes[v]['prob_sum']) == 1:
                    # equals to 1 if is the first time that the node is reached by someone
                    self.__add_node_to_the_stack_prob_sum__(v)
            elif not cim.is_active(v, self.sim_graph):
                self.sim_graph.nodes[v]['prob_sum'][agent_name] = self.sim_graph.nodes[v]['prob_sum'].get(agent_name, 0) + attr['p']
                if len(graph.nodes[v]['prob_sum']) == 1:
                    # equals to 1 if is the first time that the node is reached by someone
                    self.__add_node_to_the_stack_prob_sum__(v)

    def __add_node_to_the_stack_prob_sum__(self, node):
        self.sim_graph.graph['stack_prob_sum'].add(node)

    def __activate_nodes_in_seed_sets__(self,graph, agents):
        """
        Activate the nodes in the seed sets of the agents in the simulation graph
        """
        for agent in agents:
            for u in agent.seed:
                if not self.sim_graph.has_node(u):
                    self.__add_node__(graph, u)
                cim.activate_node(self.sim_graph, u, agent)
                self.__add_node_to_the_stack__(u)
                self.__update_prob_sum__(graph, u, agent.name)

    def __reverse_operations__(self):
        # Reset the prob_sum of the nodes that have been activated
        super().__reverse_operations__()
        stack_prob_sum = self.sim_graph.graph['stack_prob_sum']
        while stack_prob_sum:
            node = stack_prob_sum.pop()
            self.sim_graph.nodes[node]['prob_sum'] = dict()

    def activate(self, graph, agents):
        if self.sim_graph is None:
            self.__initialize_sim_graph__(graph,agents)
        self.__activate_nodes_in_seed_sets__(graph,agents)
        active_set = cim.active_nodes(self.sim_graph)
        newly_activated = list(active_set)
        while len(newly_activated)>0:
            pending_nodes = []
            for u in newly_activated:
                curr_agent_name = self.sim_graph.nodes[u]['agent'].name
                inactive_out_edges = self.__build_inactive_out_edges__(graph, u)
                for _, v, attr in inactive_out_edges:
                    if self.sim_graph.nodes[v]['prob_sum'][curr_agent_name] >= self.sim_graph.nodes[v]['threshold']:
                        cim.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        if v not in pending_nodes:
                            pending_nodes.append(v)
            # Second phase: contacted inactive nodes choose which agent to endorse by a strategy
            self.__extend_stack__(pending_nodes)
            newly_activated = cim.manage_pending_nodes(self.sim_graph, self.endorsement_policy, pending_nodes)
            active_set.extend(newly_activated)
            for u in newly_activated:
                self.__update_prob_sum__(graph, u, self.sim_graph.nodes[u]['agent'].name)
        result = self.__group_by_agent__(self.sim_graph, active_set)
        self.__reverse_operations__()
        return result