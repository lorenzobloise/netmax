import logging
import influence_maximization as im
import networkx as nx
import copy

class DiffusionModel:

    def __init__(self, endorsement_policy):
        self.endorsement_policy = endorsement_policy
        self.sim_graph = None
        self.graph_nodes = None
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.history = {} # Dictionary <iteration_id: (active_set, pending_set)> where every node has its attributes
        self.iteration_id = 0
        self.last_active_set = None

    def __copy__(self):
        result = DiffusionModel(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items():  # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def get_history(self):
        return self.history

    def __register_history__(self, active_set, pending_set):
        if active_set is not None:
            active_set = self.__group_active_set_by_agent__(active_set)
            self.last_active_set = copy.deepcopy(active_set)
        pending_set = self.__build_pending_set_for_history__(pending_set)
        self.history[self.iteration_id] = (self.last_active_set, pending_set)
        self.iteration_id += 1

    def __reset_parameters__(self):
        self.last_active_set = None
        self.history = {}
        self.iteration_id = 0

    def __add_node_to_the_stack__(self, node):
        self.sim_graph.graph['stack_active_nodes'].add(node)

    def __extend_stack__(self, nodes):
        self.sim_graph.graph['stack_active_nodes'].update(nodes)

    def __add_node__(self, graph, u):
        """
        Add a node to the simulation graph and copy its attributes from the original graph
        """
        if self.graph_nodes==None:
            self.graph_nodes = graph.nodes(data=True)
        dict_attr = self.graph_nodes[u]
        self.sim_graph.add_node(u, **dict_attr)

    def __initialize_sim_graph__(self, graph, agents):
        self.sim_graph = nx.DiGraph()
        for key, value in graph.graph.items():  # Copy the graph's attributes
            self.sim_graph.graph[key] = value
        for agent in agents:
            for u in agent.seed:
                self.__add_node__(graph, u)
        self.sim_graph.graph['stack_active_nodes'] = set()
        self.sim_graph.graph['stack_inf_prob'] = set()

    def __reverse_operations__(self, graph):
        """
        This method empties the stack of the active nodes
        """
        stack_active_nodes = self.sim_graph.graph['stack_active_nodes']
        while len(stack_active_nodes) > 0:
            node = stack_active_nodes.pop()
            im.deactivate_node_in_simulation_graph(graph,self.sim_graph, node)

    def __build_inactive_out_edges__(self, graph, u):
        inactive_out_edges = []
        for (_, v, attr) in graph.out_edges(u, data=True):
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

    def preprocess_data(self, graph):
        raise NotImplementedError("This method must be implemented by subclasses")

    def activate(self, graph, agents):
        raise NotImplementedError("This method must be implemented by subclasses")

    def __group_active_set_by_agent__(self, active_set):
        dict_result = {}
        for u in active_set:
            curr_agent = self.sim_graph.nodes[u]['agent'].name
            if curr_agent in dict_result:
                dict_result[curr_agent].append(u)
            else:
                dict_result[curr_agent] = [u]
        return dict_result

    def __build_pending_set_for_history__(self, pending_set):
        dict_result = {}
        for u in pending_set:
            contacted_by = self.sim_graph.nodes[u]['contacted_by']
            dict_result[u] = copy.deepcopy(contacted_by)
        return dict_result