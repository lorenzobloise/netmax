from multi_agent.diffusion_models.diffusion_model import DiffusionModel
import multi_agent.competitive_influence_maximization as cim
import random
import networkx as nx

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
            for key, value in self.sim_graph.graph.items():  # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def preprocess_data(self, graph):
        return

    def __add_node__(self, graph, u):
        """
        Add a node and its neighborhood to the simulation graph and copy its attributes from the original graph
        """
        dict_attr = graph.nodes(data=True)[u]
        self.sim_graph.add_node(u, **dict_attr)

    def __add_out_neighbors__(self, graph, u, sim_graph):
        """
        Add the out-neighbors of a node to the simulation graph
        """
        for (_, v, attr) in graph.out_edges(u, data=True):
            if not sim_graph.has_node(v):
                dict_attr = graph.nodes(data=True)[v]
                sim_graph.add_node(v, **dict_attr)
            if not sim_graph.has_edge(u, v):
                sim_graph.add_edge(u, v, **attr)

    def __initialize_sim_graph__(self, graph, agents):
        self.sim_graph = nx.DiGraph()
        for key, value in graph.graph.items():  # Copy the graph's attributes
            self.sim_graph.graph[key] = value
        for agent in agents:
            for u in agent.seed:
                self.__add_node__(graph, u)
        self.sim_graph.graph['stack_active_nodes'] = []
        self.sim_graph.graph['stack_inf_prob'] = [] # Stack for dynamic probabilities

    def __reverse_operations__(self):
        """
        This method empties the stack of the active nodes
        """
        stack_active_nodes = self.sim_graph.graph['stack_active_nodes']
        while len(stack_active_nodes) > 0:
            node = stack_active_nodes.pop()
            cim.deactivate_node(self.sim_graph, node)

    def __add_node_to_the_stack__(self, node):
        self.sim_graph.graph['stack_active_nodes'].append(node)

    def __extend_stack__(self, nodes):
        self.sim_graph.graph['stack_active_nodes'].extend(nodes)

    def activate(self, graph, agents):
        if self.sim_graph is None:
            self.__initialize_sim_graph__(graph, agents)
        active_set = []
        for agent in agents:
            for node in agent.seed:
                if not self.sim_graph.has_node(node):
                    self.__add_node__(graph, node)
                cim.activate_node(self.sim_graph, node, agent)
                self.__add_node_to_the_stack__(node)
                active_set.append(node)
        newly_activated = list(active_set)
        while len(newly_activated) > 0:
            # First phase: try to influence inactive nodes
            # Each newly activated node tries to activate its inactive neighbors by contacting them
            pending_nodes = []
            for u in newly_activated:
                inactive_out_edges=[]
                #inactive_out_edges = [(u, v, attr) for (u, v, attr) in graph.out_edges(u, data=True)
                #                  if sim_graph.has_node(v) and not cim.is_active(v, sim_graph) else True]
                for (_, v, attr) in graph.out_edges(u, data=True):
                    if not self.sim_graph.has_node(v):
                        inactive_out_edges.append((u, v, attr))
                    elif not cim.is_active(v, self.sim_graph):
                        inactive_out_edges.append((u, v, attr))
                for (_, v, attr) in inactive_out_edges:
                    r = random.random()
                    if r < attr['p']:
                        if not self.sim_graph.has_node(v):
                            nodes_attr = graph.nodes(data=True)[v]
                            self.sim_graph.add_node(v, **nodes_attr)
                        self.sim_graph.add_edge(u, v, **attr)
                        cim.contact_node(self.sim_graph, v, self.sim_graph.nodes[u]['agent'])
                        if v not in pending_nodes:
                            pending_nodes.append(v)
                        #self.__add_out_neighbors__(graph, v, sim_graph)
                        #if not sim_graph.has_edge(v, _): self.__add_out_neighbors__(graph, v, sim_graph)
                        #if len(sim_graph.out_edges(v))==0: self.__add_out_neighbors__(graph, v, sim_graph)
            self.__extend_stack__(pending_nodes)
            newly_activated = cim.manage_pending_nodes(self.sim_graph, self.endorsement_policy, pending_nodes)
            active_set.extend(newly_activated)
        result = self.__group_by_agent__(self.sim_graph, active_set)
        self.__reverse_operations__()
        return result