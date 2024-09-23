import copy
from algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
from heapdict import heapdict
import competitive_influence_maximization as cim
from tqdm import tqdm

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
            result.prev_best = None if self.prev_best is None else self.prev_best.__deepcopy__()
            return result

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.queues = None
        self.idx = 0
        self.last_seed = {}
        self.curr_best = {}

    def __initialize_queues__(self, sim_graph, agents_copy):
        self.queues = {self.curr_agent_id: heapdict()}
        for node in tqdm(sim_graph.nodes, desc="Initializing queues"):
            node_data=CELF_PP.Node(node)
            node_data.mg1 = self.__do_simulation__(sim_graph, agents_copy,[node_data.node])
            node_data.prev_best = None if self.curr_agent_id not in self.curr_best else self.curr_best[self.curr_agent_id]
            node_data.flag = 0
            if self.curr_agent_id in self.curr_best.keys() and self.__get_curr_best__().mg1 > node_data.mg1:
                self.curr_best[self.curr_agent_id] = self.curr_best[self.curr_agent_id]
            else:
                self.curr_best[self.curr_agent_id] = node_data
            self.__add_element_to_the_queue__(node_data)
        # In the first iteration, the first agent's queue is replicated for all agents,
        # so we make a deep copy of the queue for all agents
        for agent in self.agents:
            if agent.id == self.curr_agent_id:
                continue
            #Manage the deep copy of the curr_best
            self.curr_best[agent.id] = self.__get_curr_best__().__deepcopy__()
            #Manage the deep copy of the queue
            q_copy = heapdict()
            for node_data, neg_mg1 in list(self.queues[self.curr_agent_id].items()):
                node_data_copy = node_data.__deepcopy__()
                q_copy[node_data_copy] = neg_mg1
            self.queues[agent.id] = q_copy

    def __add_element_to_the_queue__(self, node_data):
        q = self.queues[self.curr_agent_id]
        q[node_data] = -node_data.mg1

    def __peek_top_element__(self):
        q = self.queues[self.curr_agent_id]
        node_data, neg_mg1 = q.peekitem()
        return node_data, neg_mg1

    def __remove_element_from_the_queue__(self, node_data):
        for agent in self.agents:
            curr_id = agent.id
            # Get the queue of the agent
            q = self.queues[curr_id]
            # Remove the node from the queue
            for curr_node_data in q.keys():
                if curr_node_data.node == node_data.node:
                    del q[curr_node_data]
                    break

    def __update_element_in_the_queue__(self, node_data):
        q = self.queues[self.curr_agent_id]
        q[node_data] = -node_data.mg1

    def __do_simulation__(self, sim_graph, agents, seed_set=None):
        old_seed_set = None
        if seed_set is not None:
            old_seed_set = agents[self.curr_agent_id].seed
            agents[self.curr_agent_id].seed = seed_set
        spreads: dict = cim.simulation(sim_graph, self.diff_model, agents, self.r)
        if old_seed_set is not None:
            agents[self.curr_agent_id].seed = old_seed_set
        spread_curr_agent = spreads[self.agents[self.curr_agent_id].name]
        return spread_curr_agent

    def __do_simulation_delta__(self, sim_graph, agents, seed_1, seed_2):
        """
        Run the simulation for the current agent (the agent that is currently running the algorithm)
        :return: the spread of the current agent
        """
        result: dict = cim.simulation_delta(sim_graph, self.diff_model, agents, self.curr_agent_id, seed_1, seed_2, self.r)
        spread_curr_agent = result[self.agents[self.curr_agent_id].name]
        return spread_curr_agent

    def __get_seed_set__(self, agents):
        """
        Get the seed set of the current agent (the agent that is currently running the algorithm)
        """
        return agents[self.curr_agent_id].__getattribute__('seed')

    def __get_curr_best__(self):
        """
        Get the current best node for the current agent (the agent that is currently running the algorithm)
        from the curr_best dictionary
        """
        return self.curr_best[self.curr_agent_id]

    def run(self):
        agents_copy = copy.deepcopy(self.agents)
        # If the queues are not initialized, initialize them, then do the first iteration pass of CELF++
        # TODO: manage opinions by initializing different queues for each agents instead of merely copying
        if self.queues is None:
            self.__initialize_queues__(self.graph, agents_copy)
        # Other iterations
        progress_bar = tqdm(total=self.budget, desc='Agent ' + str(self.curr_agent_id) + ' has chosen the next seed with CELF++')
        for i in range(self.budget):
            seed_added = False
            while not seed_added:
                node_data, _ = self.__peek_top_element__()
                if node_data.flag == len(agents_copy[self.curr_agent_id].seed):
                    agents_copy[self.curr_agent_id].seed.append(node_data.node)
                    agents_copy[self.curr_agent_id].spread = node_data.mg1
                    self.__remove_element_from_the_queue__(node_data)
                    self.last_seed[self.curr_agent_id] = node_data
                    seed_added = True
                    progress_bar.update(1)
                    continue
                if not node_data.mg2_already_computed:
                    node_data.mg2 = self.__do_simulation__(self.graph, agents_copy, [node_data.node] + [self.__get_curr_best__().node])
                    node_data.mg2_already_computed = True
                elif node_data.prev_best == self.last_seed[self.curr_agent_id]:
                    node_data.mg1 = node_data.mg2
                else:
                    seed_1 = self.__get_seed_set__(agents_copy) + [node_data.node]
                    seed_2 = self.__get_seed_set__(agents_copy)
                    node_data.mg1 = self.__do_simulation_delta__(self.graph, agents_copy, seed_1, seed_2)
                    node_data.prev_best = self.__get_curr_best__()
                    seed_1 = self.__get_seed_set__(agents_copy) + [self.__get_curr_best__().node] + [node_data.node]
                    seed_2 = self.__get_seed_set__(agents_copy) + [self.__get_curr_best__().node]
                    node_data.mg2 = self.__do_simulation_delta__(self.graph, agents_copy, seed_1, seed_2)
                node_data.flag = len(agents_copy[self.curr_agent_id].seed)
                if (self.curr_agent_id in self.curr_best.keys() and
                    self.__get_curr_best__().mg1 > node_data.mg1):
                    self.curr_best[self.curr_agent_id] = self.__get_curr_best__()
                else:
                    self.curr_best[self.curr_agent_id] = node_data
                self.__update_element_in_the_queue__(node_data)
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: a.spread for a in agents_copy}