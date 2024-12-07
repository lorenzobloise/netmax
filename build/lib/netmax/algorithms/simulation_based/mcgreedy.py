import copy
from netmax.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
from netmax import influence_maximization as im
from tqdm import tqdm

class MCGreedy(SimulationBasedAlgorithm):
    """
    Monte Carlo greedy works by picking iteratively the node with the maximum marginal gain until the budget is fulfilled.
    Tha marginal gains of the nodes are computed at each iteration by doing a certain number of Monte Carlo simulations
    (the typical number used in literature is 10,000). Even though the agents are more than one, we don't need to store
    different marginal gains for each agent (like we do in CELF or CELF++) because Monte Carlo greedy simply re-computes
    those value each time.
    """

    name = 'mcgreedy'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)

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
        # Repeat until the budget is fulfilled (in the InfluenceMaximization class, inside the run method, the algorithm
        # is always invoked with the budget parameter set to 1, but we preferred to write the code in a more general way)
        for _ in range(self.budget):
            marginal_gains = []
            # Store the spread of the current agent until now
            last_spread = agents_copy[self.curr_agent_id].spread
            # Examine each one of the inactive nodes and compute its marginal gain by running Monte Carlo simulations
            for u in tqdm(im.inactive_nodes(self.graph), desc='Nodes examined', leave=None):
                # Temporarily add the node u to the seed set
                agents_copy[self.curr_agent_id].seed = agents_copy[self.curr_agent_id].seed + [u]
                # Do the simulations
                spreads = im.simulation(self.graph, self.diff_model, agents=agents_copy, r=self.r)
                # Compute the marginal gain as the difference between the spreads
                marginal_gain = spreads[self.agents[self.curr_agent_id].name] - last_spread
                # Store the marginal gain in the dictionary
                marginal_gains.append((u, marginal_gain, spreads))
                # We appended the node to the seed set, now we remove it
                agents_copy[self.curr_agent_id].seed = agents_copy[self.curr_agent_id].seed[:-1]
            # Select the node with the maximum marginal gain
            u, top_gain, spreads = max(marginal_gains, key=lambda x: x[1])
            self.__update_spreads__(agents_copy, spreads)
            # Update the agent's seed and spread
            agents_copy[self.curr_agent_id].seed.append(u)
            im.activate_node(self.graph, u, self.agents[self.curr_agent_id]) # Activate the top node
        # Return the new nodes to add to the seed set and the spread
        result_seed_set = agents_copy[self.curr_agent_id].seed[-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: a.spread for a in agents_copy}