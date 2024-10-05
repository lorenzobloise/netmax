from heapdict import heapdict
from algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
import copy
import influence_maximization as im
from tqdm import tqdm
from influence_probabilities import OpinionBased

class CELF(SimulationBasedAlgorithm):

    name = 'celf'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.queues = {agent.id: heapdict() for agent in self.agents} # Dictionary {agent_id: queue}

    def __first_monte_carlo__(self, graph, agents):
        """
        :return: dictionary of marginal gains of each node sorted in descending order by marginal gain
        """
        for u in tqdm(im.inactive_nodes(graph), desc="Choosing first node and initializing queues"):
            agents[self.curr_agent_id].seed.append(u)
            spreads = im.simulation(graph=graph, diff_model=self.diff_model, agents=agents, r=self.r)
            spread_value = spreads[self.agents[self.curr_agent_id].name]
            if self.graph.graph['inf_prob'].__class__ != OpinionBased and len(agents[self.curr_agent_id].seed) == 1:
                # Generate the same queue for each agent
                for a in agents:
                    # We did the simulation from the perspective of the current agent
                    # But this is the first simulation, so the seed set is empty
                    # and the spread value is equal for all agents
                    q = self.queues[a.id]
                    q[u] = -spread_value
            else: # Generate only the current agent's queue
                q = self.queues[self.curr_agent_id]
                q[u] = -spread_value
            # We appended the node to the seed set, now we remove it
            agents[self.curr_agent_id].seed = agents[self.curr_agent_id].seed[:-1]

    def __pop_top_node_and_marginal_gain__(self):
        """
        Take the top node and its marginal gain from the queue of the current agent
        and remove it from the queues of the other agents
        """
        top_node, neg_top_marginal_gain = self.queues[self.curr_agent_id].popitem()
        top_marginal_gain = -neg_top_marginal_gain
        self.__remove_node_from_queues__(top_node)
        return top_node, top_marginal_gain

    def __remove_node_from_queues__(self, node):
        for agent in self.agents:
            q = self.queues[agent.id]
            for curr_node in list(q.keys()):
                if curr_node == node:
                    del q[curr_node]

    def __peek_top_node_and_marginal_gain__(self):
        """
        Peek the top node and its marginal gain from the queue of the current agent
        """
        top_node, neg_top_marginal_gain = self.queues[self.curr_agent_id].peekitem()
        top_marginal_gain = -neg_top_marginal_gain
        return top_node, top_marginal_gain

    def __update_queue_of_the_current_agent__(self, u, new_marg_gain):
        q = self.queues[self.curr_agent_id]
        q[u] = -new_marg_gain

    def __get_marginal_gain_of_u__(self, graph, agents, u, last_spread):
        """
        :return: marginal gain of node u
        """
        agents[self.curr_agent_id].seed = agents[self.curr_agent_id].seed + [u]
        spreads = im.simulation(graph, self.diff_model, agents, self.r)
        curr_marg_gain = spreads[self.agents[self.curr_agent_id].name] - last_spread
        agents[self.curr_agent_id].seed = agents[self.curr_agent_id].seed[:-1]
        return curr_marg_gain, spreads

    def __update_spreads__(self, agents, spreads):
        for agent in agents:
            if agent.name in spreads:
                agent.spread = spreads[agent.name]
            else:
                agent.spread = 0

    def run(self):
        agents_copy = copy.deepcopy(self.agents)
        if len(self.queues[self.curr_agent_id]) == 0:
            self.__first_monte_carlo__(graph=self.graph, agents=agents_copy)
            top_node, top_marginal_gain = self.__pop_top_node_and_marginal_gain__()
            spreads = {}
            for agent in agents_copy:
                spreads[agent.name] = top_marginal_gain if agent.id == self.curr_agent_id else 0
            return [top_node], spreads
        for _ in range(self.budget):
            check = False
            last_spread = agents_copy[self.curr_agent_id].spread
            updated_spreads = None
            while not check:
                u, _ = self.__peek_top_node_and_marginal_gain__()
                # Do a simulation with the new seed set
                curr_marg_gain, spreads_sim = self.__get_marginal_gain_of_u__(self.graph, agents_copy, u, last_spread)
                self.__update_queue_of_the_current_agent__(u, curr_marg_gain)
                updated_spreads = spreads_sim
                curr_top_node, _ = self.__peek_top_node_and_marginal_gain__()
                if curr_top_node == u:
                    check = True
            top_node, top_marginal_gain = self.__pop_top_node_and_marginal_gain__()
            self.__update_spreads__(agents_copy, updated_spreads)
            agents_copy[self.curr_agent_id].seed.append(top_node)
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: a.spread for a in agents_copy}