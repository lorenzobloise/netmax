from heapdict import heapdict
from netmax.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
import copy
from netmax import influence_maximization as im
from tqdm import tqdm
from netmax.influence_probabilities import OpinionBased

class CELF(SimulationBasedAlgorithm):
    """
    Paper: Leskovec et al. - "Cost-Effective Outbreak Detection in Networks"
    CELF improves the Monte Carlo greedy algorithm by maintaining a priority queue sorted in descending order based on the
    marginal gain of the nodes. Since we have more than one agent, we also need more than one queue, specifically one
    for each agent.
    """

    name = 'celf'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        # Dictionary {agent_id: queue}, where heapdict() is a min-heap, so when we're storing the marginal gains we have
        # to insert negative values in order to have the best node at the beginning of the queue
        self.queues = {agent.id: heapdict() for agent in self.agents}

    def __first_monte_carlo__(self, graph, agents_copy):
        """
        Does a classic Monte Carlo simulation, like the ones done inside the greedy algorithm.
        :param graph: The graph.
        :param agents_copy: The deep copy of the 'agents' dictionary.
        :return: dictionary of marginal gains of each node sorted in descending order based on the marginal gain
        """
        # Iterate through all inactive nodes
        for u in tqdm(im.inactive_nodes(graph), desc="Choosing first node and initializing queues"):
            # Temporarily add the node u to the seed set
            agents_copy[self.curr_agent_id].seed.append(u)
            # Do the simulations
            spreads = im.simulation(graph=graph, diff_model=self.diff_model, agents=agents_copy, r=self.r)
            spread_value = spreads[self.agents[self.curr_agent_id].name]
            # If we're not using the opinion-based influence probability, the first Monte Carlo simulation is valid
            # for all the agents, since the nodes do not have an exogenous information (i.e. the opinion)
            # Otherwise, if we're using the opinion-based influence probability, we have to replicate the
            # first Monte Carlo simulation for each agent
            if self.graph.graph['inf_prob'].__class__ != OpinionBased and len(agents_copy[self.curr_agent_id].seed) == 1:
                # Generate the same queue for each agent
                for a in agents_copy:
                    # We did the simulation from the perspective of the current agent, but this is the first simulation,
                    # so the seed set is empty and the spread value is equal for all agents
                    q = self.queues[a.id]
                    q[u] = -spread_value # Is negative to implement the decreasing order of the queue
            else: # Generate only the current agent's queue
                q = self.queues[self.curr_agent_id]
                q[u] = -spread_value # Is negative to implement the decreasing order of the queue
            # We appended the node to the seed set, now we remove it
            agents_copy[self.curr_agent_id].seed = agents_copy[self.curr_agent_id].seed[:-1]

    def __pop_top_node_and_marginal_gain__(self):
        """
        Take the top node and its marginal gain from the queue of the current agent and remove it also from the queues of the other agents.
        :return: the top node and its marginal gain.
        """
        top_node, neg_top_marginal_gain = self.queues[self.curr_agent_id].popitem()
        top_marginal_gain = -neg_top_marginal_gain # Originally neg_top_marginal_gain is a negative number, so we make it positive
        self.__remove_node_from_queues__(top_node) # Remove the node from other queues
        return top_node, top_marginal_gain

    def __remove_node_from_queues__(self, node):
        """
        Removes the node from all the queues.
        :param node: The node to remove.
        """
        for agent in self.agents:
            # Take the queue of the agent
            q = self.queues[agent.id]
            for curr_node in list(q.keys()):
                # Remove the node
                if curr_node == node:
                    del q[curr_node]

    def __peek_top_node_and_marginal_gain__(self):
        """
        Peek the top node and its marginal gain from the queue of the current agent.
        :return: the top node and its marginal gain.
        """
        top_node, neg_top_marginal_gain = self.queues[self.curr_agent_id].peekitem()
        top_marginal_gain = -neg_top_marginal_gain # Originally neg_top_marginal_gain is a negative number, so we make it positive
        return top_node, top_marginal_gain

    def __update_queue_of_the_current_agent__(self, u, new_marg_gain):
        """
        Updates the queue of the current agent by setting the new marginal gain of the node u.
        :param u: The node to update.
        :param new_marg_gain: The new marginal gain of the node u.
        """
        q = self.queues[self.curr_agent_id]
        q[u] = -new_marg_gain

    def __get_marginal_gain_of_u__(self, graph, agents_copy, u, last_spread):
        """
        :param graph: The graph.
        :param agents_copy: The deep copy of the 'agents' dictionary.
        :param u: The node we want the marginal gain of.
        :param last_spread: The last spread of the current agent.
        :return: the marginal gain of the node u.
        """
        # Temporarily add the node u to the seed set
        agents_copy[self.curr_agent_id].seed = agents_copy[self.curr_agent_id].seed + [u]
        # Do the simulations
        spreads = im.simulation(graph, self.diff_model, agents_copy, self.r)
        # Compute the current marginal gain of the node u
        curr_marg_gain = spreads[self.agents[self.curr_agent_id].name] - last_spread
        # We appended the node to the seed set, now we remove it
        agents_copy[self.curr_agent_id].seed = agents_copy[self.curr_agent_id].seed[:-1]
        return curr_marg_gain, spreads

    def __update_spreads__(self, agents_copy, spreads):
        """
        Updates the agents spread after selecting the node with the maximum marginal gain.
        :param agents_copy: The deep copy of the 'agents' dictionary.
        :param spreads: The 'spreads' dictionary, result of the simulations.
        """
        for agent in agents_copy:
            if agent.name in spreads:
                agent.spread = spreads[agent.name]
            else:
                agent.spread = 0

    def run(self):
        """
        :return: The nodes to add in the seed set of the current agent and the spreads for each agent.
        """
        # Make a deep copy of the 'agents' dictionary
        agents_copy = copy.deepcopy(self.agents)
        # If it's the first iteration, initialize the queues with the result of a Monte Carlo simulation
        if len(self.queues[self.curr_agent_id]) == 0:
            self.__first_monte_carlo__(graph=self.graph, agents_copy=agents_copy)
            # Take the top node and its marginal gain from the current agent's queue
            top_node, top_marginal_gain = self.__pop_top_node_and_marginal_gain__()
            # Set the spreads for each agent
            spreads = {}
            for agent in agents_copy:
                spreads[agent.name] = top_marginal_gain if agent.id == self.curr_agent_id else 0
            return [top_node], spreads
        # Repeat until the budget is fulfilled (in the InfluenceMaximization class, inside the run method, the algorithm
        # is always invoked with the budget parameter set to 1, but we preferred to write the code in a more general way)
        for _ in range(self.budget):
            check = False # Boolean to tell if a node must be chosen or not
            last_spread = agents_copy[self.curr_agent_id].spread
            updated_spreads = None
            # Iterate until we find a node suitable for being chosen as part of the seed set of the current agent:
            # - peek the top node
            # - do a simulation and update the marginal gains
            # - if it's still the top node, choose it, otherwise repeat indefinitely
            while not check:
                u, _ = self.__peek_top_node_and_marginal_gain__()
                # Do a simulation with the new seed set
                curr_marg_gain, spreads_sim = self.__get_marginal_gain_of_u__(self.graph, agents_copy, u, last_spread)
                # Update the queue
                self.__update_queue_of_the_current_agent__(u, curr_marg_gain)
                updated_spreads = spreads_sim
                # Check if it's still the top node of the queue, and if it is then set 'check' to True to exit the loop
                curr_top_node, _ = self.__peek_top_node_and_marginal_gain__()
                if curr_top_node == u:
                    check = True
            # Remove the top node from all the queues
            top_node, top_marginal_gain = self.__pop_top_node_and_marginal_gain__()
            # Update the spreads
            self.__update_spreads__(agents_copy, updated_spreads)
            agents_copy[self.curr_agent_id].seed.append(top_node)
        # Return the new nodes to add to the seed set and the spread
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: a.spread for a in agents_copy}