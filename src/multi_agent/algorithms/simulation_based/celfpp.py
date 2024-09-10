
from src.multi_agent.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
from heapdict import heapdict
import src.multi_agent.competitive_influence_maximization as cim

class CELF_PP(SimulationBasedAlgorithm):
    """
    Paper: Goyal et al. - "CELF++: Optimizing the Greedy Algorithm for Influence Maximization in Social Networks"
    """

    name = 'celfpp'

    class Node(object):

        _idx = 0

        def __init__(self, node):
            self.node = node
            self.mg1 = 0
            self.prev_best = None
            self.mg2 = 0
            self.mg2_already_computed = False
            self.flag = None
            self.id = CELF_PP.Node._idx
            CELF_PP.Node._idx += 1

        def __hash__(self):
            return self.id

        def __deepcopy__(self):
            result = CELF_PP.Node(self.node)
            result.mg1 = self.mg1
            result.mg2 = self.mg2
            result.mg2_already_computed = self.mg2_already_computed
            result.flag = self.flag
            result.prev_best = self.prev_best._deepcopy__()
            return result

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.queues = None
        self.idx = 0
        self.last_seed = {}
        self.curr_best = {}

    def __initialize_queues__(self, sim_graph, agents_copy):
        self.queues = {}
        Q = heapdict()  # Priority Queue based on marginal gain 1
        for node in sim_graph.nodes:
            node_data=CELF_PP.Node(node)
            agents_copy[self.curr_agent_id].seed = [node_data.node]
            node_data.mg1 = cim.simulation(sim_graph, self.diff_model, agents_copy, self.r)
            agents_copy[self.curr_agent_id].seed = []
            node_data.prev_best = None if self.curr_agent_id not in self.curr_best else self.curr_best[self.curr_agent_id]
            node_data.flag = 0
            if self.curr_agent_id in self.curr_best.keys() and self.curr_best[self.curr_agent_id].mg1 > node_data.mg1:
                self.curr_best[self.curr_agent_id] = self.curr_best[self.curr_agent_id]
            else:
                self.curr_best[self.curr_agent_id] = node_data
            self.__add_element_to_the_queue__(node_data)
        # In the first iteration, the first agent's queue is replicated for all agents,
        # so we make a deep copy of the queue for all agents
        for agent_id in self.agents:
            if agent_id == self.curr_agent_id:
                continue
            #Manage the deep copy of the curr_best
            self.curr_best[agent_id] = self.curr_best[self.curr_agent_id].__deepcopy__()
            #Manage the deep copy of the queue
            q_copy = heapdict()
            for node_data, neg_mg1 in list(Q.items()):
                node_data_copy = node_data.__deepcopy__()
                q_copy[node_data_copy] = neg_mg1
            self.queues[agent_id] = q_copy

    def __add_element_to_the_queue__(self, node_data):
        q=self.queues[self.curr_agent_id]
        q[node_data] = -node_data.mg1

    def __peek_top_element__(self):
        q = self.queues[self.curr_agent_id]
        node_data, neg_mg1 = q.peekitem()
        return node_data, neg_mg1

    def __remove_element_from_the_queue__(self,node_data):
        q = self.queues[self.curr_agent_id]
        del q[node_data]
    def __update_element_in_the_queue__(self, node_data):
        q = self.queues[self.curr_agent_id]
        q[node_data] = -node_data.mg1

    def run(self):
        sim_graph = self.graph.copy()
        agents_copy = self.agents.copy()
        # If the queues are not initialized, initialize them, then do the first iteration pass of CELF++
        if self.queues is None:
            self.__initialize_queues__(sim_graph, agents_copy)
        # Other iterations
        for i in range(self.budget):
            seed_added = False
            while not seed_added:
                node_data, _ = self.__peek_top_element__()
                if not node_data.mg2_already_computed:
                    old_seed_set = agents_copy[self.curr_agent_id].seed
                    agents_copy[self.curr_agent_id].seed = [node_data.node] + [self.curr_best[self.curr_agent_id].node]
                    node_data.mg2 = cim.simulation(sim_graph, self.diff_model, agents_copy, self.r)
                    node_data.mg2_already_computed = True
                    agents_copy[self.curr_agent_id].seed = old_seed_set
                if node_data.flag == len(agents_copy[self.curr_agent_id].seed):
                    agents_copy[self.curr_agent_id].seed.append(node_data.node)
                    self.__remove_element_from_the_queue__(node_data)
                    self.last_seed[self.curr_agent_id] = node_data
                    seed_added = True
                    continue
                elif node_data.prev_best == self.last_seed[self.curr_agent_id]:
                    node_data.mg1 = node_data.mg2
                else:
                    node_data.mg1 = cim.simulalation_delta() # TODO: implement this
                    node_data.prev_best = self.curr_best[self.curr_agent_id]
                    node_data.mg2 = cim.simulation_delta() # TODO: implement this
                node_data.flag = len(agents_copy[self.curr_agent_id].seed)
                if (self.curr_agent_id in self.curr_best.keys() and
                    self.curr_best[self.curr_agent_id].mg1 > node_data.mg1):
                    self.curr_best[self.curr_agent_id] = self.curr_best[self.curr_agent_id]
                else:
                    self.curr_best[self.curr_agent_id] = node_data
                self.__update_element_in_the_queue__(node_data)
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, agents_copy[self.curr_agent_id].spread