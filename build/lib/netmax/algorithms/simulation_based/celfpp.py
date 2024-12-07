import copy
from netmax.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
from heapdict import heapdict
from netmax import influence_maximization as im
from tqdm import tqdm

from netmax.influence_probabilities import OpinionBased

class CELF_PP(SimulationBasedAlgorithm):
    """
    Paper: Goyal et al. - "CELF++: Optimizing the Greedy Algorithm for Influence Maximization in Social Networks"
    CELF++ diminishes the computation time of CELF by optimizing its priority queue and exploiting the submodularity
    of the influence spread function. Since we have more than one agent, we also need more than one queue, specifically one
    for each agent.
    """

    name = 'celfpp'

    class Node(object):
        """
        Class that encapsulates a node and the information associated with it, useful to the CELF++ algorithm's logic.
        """
        # Static class variable that gets incremented every time the class is instantiated to have a unique identifier
        _idx = 0

        def __init__(self, node):
            self.node = node
            self.mg1 = 0 # Marginal gain of the node w.r.t. the current seed set
            # prev_best is the node that has the maximum marginal gain among all the ones examined in the current iteration before the current node.
            # The idea is that if the node prev_best is picked as a seed in the current iteration, we don't need to recompute
            # the marginal gain of this node w.r.t. the current seed set with prev_best added to it in the next iteration (Goyal et al.)
            self.prev_best = None
            self.mg2 = 0 # Marginal gain of the node w.r.t. the current seed set with prev_best added to it
            self.mg2_already_computed = False # True if mg2 is already computed for this node
            self.flag = None # Iteration number when mg1 was last updated
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
        self.queues = {}
        self.idx = 0
        self.last_seed = {}
        self.curr_best = {} # The current best is different for each agent, so this is a dictionary <agent_id: curr_best_node>

    def __initialize_queues__(self, graph, agents_copy):
        """
        Initializes the priority queues for each agent.
        :param graph: The graph.
        :param agents_copy: The deep copy of the 'agents' dictionary.
        """
        # Initialize the current agent's queue
        self.queues[self.curr_agent_id] = heapdict()
        # Iterate through the inactive nodes
        for node in tqdm(im.inactive_nodes(graph), desc="Initializing queues"):
            node_data = CELF_PP.Node(node)
            # Compute mg1 by doing a simulation with the current seed set
            node_data.mg1 = self.__do_simulation__(graph, agents_copy,[node_data.node])
            # This node's prev_best is the curr_best of the current agent, or None if the current agent doesn't have a curr_best
            node_data.prev_best = None if self.curr_agent_id not in self.curr_best else self.curr_best[self.curr_agent_id]
            # Initialize the flag as this iteration number
            node_data.flag = 0
            # Update curr_best based on mg1: if the current agent has a curr_best and the mg1 of the curr_best is
            # greater than this node's mg1, don't do anything, otherwise update the current agent's curr_best with this node
            if self.curr_agent_id in self.curr_best.keys() and self.__get_curr_best__().mg1 > node_data.mg1:
                self.curr_best[self.curr_agent_id] = self.curr_best[self.curr_agent_id]
            else:
                self.curr_best[self.curr_agent_id] = node_data
            # Add the node to the queue
            self.__add_element_to_the_queue__(node_data)
        # If we're not using the opinion-based influence probability, the queue initialization is valid for all the
        # agents, since the nodes do not have an exogenous information (i.e. the opinion). Otherwise, if we're using
        # the opinion-based influence probability, we have to replicate the queue initialization for each agent
        if self.graph.graph['inf_prob'].__class__ != OpinionBased and len(agents_copy[self.curr_agent_id].seed) == 0:
            # If these conditions are satisfied, replicate the first agent's queue for each agent
            for agent in self.agents:
                if agent.id == self.curr_agent_id:
                    continue
                # Make the deep copy of the curr_best
                self.curr_best[agent.id] = self.__get_curr_best__().__deepcopy__()
                # Make the deep copy of the queue
                q_copy = heapdict()
                for node_data, neg_mg1 in list(self.queues[self.curr_agent_id].items()):
                    node_data_copy = node_data.__deepcopy__()
                    q_copy[node_data_copy] = neg_mg1
                self.queues[agent.id] = q_copy

    def __add_element_to_the_queue__(self, node_data):
        """
        Add the node to the queue.
        :param node_data: The node to be added.
        """
        q = self.queues[self.curr_agent_id]
        q[node_data] = -node_data.mg1 # Is negative to implement the decreasing order of the queue

    def __peek_top_element__(self):
        """
        Peeks the top element from the queue and returns it along with its marginal gain.
        :return: the top node from the queue and its marginal gain.
        """
        q = self.queues[self.curr_agent_id]
        node_data, neg_mg1 = q.peekitem()
        return node_data, neg_mg1

    def __remove_element_from_the_queues__(self, node_data):
        """
        Removes the node from the queues.
        :param node_data: The node to be removed.
        """
        for agent in self.agents:
            curr_id = agent.id
            # If this agent doesn't have its queue instantiated yet, don't do anything
            if not curr_id in self.queues:
                continue
            # Otherwise, get the queue of the agent
            q = self.queues[curr_id]
            # Remove the node from the queue
            for curr_node_data in q.keys():
                if curr_node_data.node == node_data.node:
                    del q[curr_node_data]
                    break

    def __update_element_in_the_queue__(self, node_data):
        """
        Updates the marginal gain of the node in the queue.
        :param node_data: The node which marginal gain has to be updated.
        """
        q = self.queues[self.curr_agent_id]
        q[node_data] = -node_data.mg1 # Is negative to implement the decreasing order of the queue

    def __do_simulation__(self, graph, agents_copy, seed_set=None):
        """
        Does a simulation to estimate the spread of the current agent
        :param graph: The graph.
        :param agents_copy: The deep copy of the 'agents' dictionary.
        :param seed_set: The seed set we want to estimate the spread of. If None, the simulation is done with the current seed set.
        :return: the estimated spread of the current agent.
        """
        old_seed_set = None
        if seed_set is not None:
            # Store the old seed set of the current agent and update with the new one (if it's not None)
            old_seed_set = agents_copy[self.curr_agent_id].seed
            agents_copy[self.curr_agent_id].seed = seed_set
        # Estimate the spreads by doing a simulation
        spreads: dict = im.simulation(graph, self.diff_model, agents_copy, self.r)
        # If the old seed set has been stored, it means that the simulation has been done with another seed set
        # (passed as a parameter), so the old seed set must be restored
        if old_seed_set is not None:
            agents_copy[self.curr_agent_id].seed = old_seed_set
        # Compute the spread of the current agent
        spread_curr_agent = spreads[self.agents[self.curr_agent_id].name]
        return spread_curr_agent

    def __do_simulation_delta__(self, graph, agents_copy, seed_1, seed_2):
        """
        Runs for the current agent the simulation_delta method of the InfluenceMaximization class (see documentation),
        then returns the estimated spread of the current agent.
        :param graph: The graph.
        :param agents_copy: The deep copy of the 'agents' dictionary.
        :param seed_1: The first seed set.
        :param seed_2: The second seed set.
        :return: the spread of the current agent.
        """
        # Invoke the method of the InfluenceMaximization class
        result: dict = im.simulation_delta(graph, self.diff_model, agents_copy, self.curr_agent_id, seed_1, seed_2, self.r)
        # Extract the spread of the current agent and return it
        spread_curr_agent = result[self.agents[self.curr_agent_id].name]
        return spread_curr_agent

    def __get_seed_set__(self, agents_copy):
        """
        Gets the seed set of the current agent.
        :param agents_copy: The deep copy of the 'agents' dictionary.
        :return: the seed set of the current agent.
        """
        return agents_copy[self.curr_agent_id].__getattribute__('seed')

    def __get_curr_best__(self):
        """
        Gets the current best node for the current agent from the curr_best dictionary.
        :return: the current best node for the current agent.
        """
        return self.curr_best[self.curr_agent_id]

    def run(self):
        """
        :return: The nodes to add in the seed set of the current agent and the spreads for each agent.
        """
        # Make a deep copy of the 'agents' dictionary
        agents_copy = copy.deepcopy(self.agents)
        # If the queues are not initialized, initialize them and do the first iteration pass of CELF++
        if self.curr_agent_id not in self.queues:
            self.__initialize_queues__(self.graph, agents_copy)
        # Other iterations of CELF++
        progress_bar = tqdm(total=self.budget, desc='Choosing the next node')
        # Repeat until the budget is fulfilled (in the InfluenceMaximization class, inside the run method, the algorithm
        # is always invoked with the budget parameter set to 1, but we preferred to write the code in a more general way)
        for i in range(self.budget):
            seed_added = False # Boolean to tell if a node has been chosen or not
            while not seed_added:
                # Peek the top element from the queue of the current agent
                node_data, _ = self.__peek_top_element__()
                # If the flag is equal to the length of the seed set, it means that in the last iteration
                # the mg1 of this node was updated, so we add this node as part of the seed set
                if node_data.flag == len(agents_copy[self.curr_agent_id].seed):
                    agents_copy[self.curr_agent_id].seed.append(node_data.node)
                    agents_copy[self.curr_agent_id].spread = node_data.mg1
                    # Remove the node from the queues
                    self.__remove_element_from_the_queues__(node_data)
                    self.last_seed[self.curr_agent_id] = node_data
                    seed_added = True
                    progress_bar.update(1)
                    continue
                # Optimization not present in the original paper, made to compute mg2 a restricted number of times
                # and thus reducing the execution time
                if not node_data.mg2_already_computed:
                    # If the mg2 has not been computed yet, do a simulation with the current seed set with the
                    # current best added to it and estimate the spread
                    node_data.mg2 = self.__do_simulation__(self.graph, agents_copy, [node_data.node] + [self.__get_curr_best__().node])
                    node_data.mg2_already_computed = True
                # If the mg2 has been already computed and the prev_best of the node is equal to the last node
                # the current agent has chosen as part of his seed set, update this node's mg1 with his own mg2
                elif node_data.prev_best == self.last_seed[self.curr_agent_id]:
                    node_data.mg1 = node_data.mg2
                # If none of the previous conditions have occurred, estimate mg1 and mg2 by invoking the method
                # simulation_delta of the InfluenceMaximization class two times:
                # 1) The first seed set is the current one with the current node added to it, and the second is just
                #    the current seed set
                # 2) The first seed set is the current one with the curr_best and current node added to it, and the
                #    second is the current seed set with the curr_best added to it
                else:
                    seed_1 = self.__get_seed_set__(agents_copy) + [node_data.node]
                    seed_2 = self.__get_seed_set__(agents_copy)
                    node_data.mg1 = self.__do_simulation_delta__(self.graph, agents_copy, seed_1, seed_2)
                    # Also, update this node's prev_best with the curr_best of the current agent
                    node_data.prev_best = self.__get_curr_best__()
                    seed_1 = self.__get_seed_set__(agents_copy) + [self.__get_curr_best__().node] + [node_data.node]
                    seed_2 = self.__get_seed_set__(agents_copy) + [self.__get_curr_best__().node]
                    node_data.mg2 = self.__do_simulation_delta__(self.graph, agents_copy, seed_1, seed_2)
                # Update this node's flag
                node_data.flag = len(agents_copy[self.curr_agent_id].seed)
                # Update curr_best based on mg1
                if (self.curr_agent_id in self.curr_best.keys() and
                    self.__get_curr_best__().mg1 > node_data.mg1):
                    self.curr_best[self.curr_agent_id] = self.__get_curr_best__()
                else:
                    self.curr_best[self.curr_agent_id] = node_data
                # Reinsert this node inside the queue and heapify
                self.__update_element_in_the_queue__(node_data)
        # Return the new nodes to add to the seed set and the spread
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: a.spread for a in agents_copy}