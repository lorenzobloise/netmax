import networkx as nx

from multi_agent.diffusion_models import IndependentCascade
from multi_agent.diffusion_models.diffusion_model import DiffusionModel
import multi_agent.competitive_influence_maximization as cim
import random

class LinearThreshold(DiffusionModel):
    """
    Paper: Granovetter et al. - "Threshold models of collective behavior"
    """

    name = 'lt'

    def __init__(self, endorsement_policy):
        super().__init__(endorsement_policy)

    def __copy__(self):
        result= LinearThreshold(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items():  # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

    def preprocess_data(self, graph):
        for node in graph.nodes:
            graph.nodes[node]['threshold'] = random.random()
            graph.nodes[node]['prob_sum'] = dict()


    def __update_prob_sum__(self, graph, node, agent_name):
        #delete the prob_sum dict of the newly activated node to avoid memory waste
        if 'prob_sum' in graph.nodes[node]:
            del graph.nodes[node]['prob_sum']
            self.__add_node_to_the_stack_prob_sum__(node)

        # I need to remember that i need to reset the prob_sum at the end of the simulation
        #for each neighbor of the newly activated node
        for (_, v, attr) in graph.out_edges(node, data=True):
            #if the neighbor is not active
            if not self.sim_graph.has_node(v) or not cim.is_active(v, self.sim_graph):
                graph.nodes[v]['prob_sum'][agent_name] = graph.nodes[v]['prob_sum'].get(agent_name, 0) + attr['p']
                if len(graph.nodes[v]['prob_sum']) == 1:
                    # equals to 1 if is the first time that the node is reached by someone
                    self.__add_node_to_the_stack_prob_sum__(v)

    #Same except for the stack_prob_sum
    def __initialize_sim_graph__(self, graph, agents):
        self.sim_graph = nx.DiGraph()
        for key, value in graph.graph.items():  # Copy the graph's attributes
            self.sim_graph.graph[key] = value
        for agent in agents:
            for u in agent.seed:
                self.__add_node__(graph, u)
        self.sim_graph.graph['stack_active_nodes'] = []
        self.sim_graph.graph['stack_inf_prob'] = [] # Stack for dynamic probabilities
        self.sim_graph.graph['stack_prob_sum'] = set() # Stack for dynamic prob_sum

    #Same
    def __add_node__(self, graph, u):
        """
        Add a node to the simulation graph and copy its attributes from the original graph
        """
        dict_attr = graph.nodes(data=True)[u]
        self.sim_graph.add_node(u, **dict_attr)

    #Same
    def __add_node_to_the_stack__(self, node):
        self.sim_graph.graph['stack_active_nodes'].append(node)

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

    def __build_inactive_out_edges__(self, graph, u):
        inactive_out_edges = []
        for (_, v, attr) in graph.out_edges(u, data=True):
            if not self.sim_graph.has_node(v):
                # If not in the simulation graph, is not active
                # because the node has not been reached yet
                inactive_out_edges.append((v, attr))
                # add the node
                nodes_attr = graph.nodes(data=True)[v]
                self.sim_graph.add_node(v, **nodes_attr)
            elif not cim.is_active(v, self.sim_graph):
                inactive_out_edges.append((v, attr))
        return inactive_out_edges


    def __reverse_operation__(self,graph):
        # Reset the prob_sum of the nodes that have been activated

        for node in self.sim_graph.graph['stack_prob_sum']:
            graph.nodes[node]['prob_sum'] = dict()
        self.sim_graph.graph['stack_prob_sum'].clear()
        # Reset the active nodes
        for node in self.sim_graph.graph['stack_active_nodes']:
            cim.deactivate_node(self.sim_graph, node)
            graph.nodes[node]['prob_sum'] = dict()
        self.sim_graph.graph['stack_active_nodes'].clear()

    def  __extend_stack__(self, nodes):
        self.sim_graph.graph['stack_active_nodes'].extend(nodes)

    def activate(self, graph, agents):
        if self.sim_graph is None:
            self.__initialize_sim_graph__(graph,agents)
        self.__activate_nodes_in_seed_sets__(graph,agents)
        active_set = cim.active_nodes(self.sim_graph)
        newly_activated = list(active_set)
        while len(newly_activated)>0:
            pending_nodes=[]
            for u in newly_activated:
                curr_agent_name = self.sim_graph.nodes[u]['agent'].name
                inactive_out_edges = self.__build_inactive_out_edges__(graph, u)
                for v, attr in inactive_out_edges:
                    if graph.nodes[v]['prob_sum'][curr_agent_name] >= graph.nodes[v]['threshold']:
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
        self.__reverse_operation__(graph)
        return result