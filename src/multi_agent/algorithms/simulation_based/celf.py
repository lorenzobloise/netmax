from src.multi_agent.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
import copy
from tqdm import tqdm
import src.multi_agent.competitive_influence_maximization as cim

class CELF(SimulationBasedAlgorithm):

    name = 'celf'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.marginal_gains = None # List of tuples (node, marginal_gain)

    def __first_monte_carlo__(self, sim_graph):
        """
        :return: dictionary of marginal gains of each node sorted in descending order by marginal gain
        """
        marginal_gains = list()
        agents_copy = copy.deepcopy(self.agents)
        for u in tqdm(cim.inactive_nodes(sim_graph), desc='First Monte Carlo simulation'):
            agents_copy[self.curr_agent_id].seed.append(u)
            spreads = cim.simulation(graph=sim_graph, diff_model=self.diff_model, agents=agents_copy, r=self.r)
            for a in self.agents:
                if a.name not in spreads.keys():
                    spreads[a.name] = list(spreads.values())[0]
            marginal_gains.append((u, spreads))
            agents_copy[self.curr_agent_id].seed = agents_copy[self.curr_agent_id].seed[:-1]
        marginal_gains = sorted(marginal_gains, key=lambda item: item[1][self.agents[self.curr_agent_id].name], reverse=True)
        return marginal_gains

    def run(self):
        sim_graph = self.graph.copy()
        if self.marginal_gains is None:
            self.marginal_gains = self.__first_monte_carlo__(sim_graph)
            top_node, top_marginal_gain = self.marginal_gains.pop(0)
            #In the first iteration the seed set is empty so the marginal gain is the spread of the agent
            return [top_node], top_marginal_gain[self.agents[self.curr_agent_id].name]
        agents_copy = copy.deepcopy(self.agents)
        for _ in range(self.budget):
            check = False
            while not check:
                u = self.marginal_gains[0][0]
                # Do a simulation with the new seed set
                curr_marg_gain = self.__get_marginal_gain_of_u__(sim_graph, agents_copy, u)
                self.marginal_gains[0][1][self.agents[self.curr_agent_id].name] = curr_marg_gain
                self.marginal_gains = sorted(self.marginal_gains, key=lambda item: item[1][self.agents[self.curr_agent_id].name], reverse=True)
                if self.marginal_gains[0][0] == u:
                    check = True
            (top_node, top_marginal_gain) = self.marginal_gains.pop(0)
            agents_copy[self.curr_agent_id].spread += top_marginal_gain[self.agents[self.curr_agent_id].name]
            agents_copy[self.curr_agent_id].seed.append(top_node)
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, agents_copy[self.curr_agent_id].spread