import copy
from multi_agent.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
import multi_agent.competitive_influence_maximization as cim
from tqdm import tqdm

class MCGreedy(SimulationBasedAlgorithm):

    name = 'mcgreedy'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)

    def run(self):
        sim_graph = self.graph.copy()
        agents_copy = copy.deepcopy(self.agents)
        for _ in tqdm(range(self.budget), desc=f'Agent {self.curr_agent_id} choosing node', leave=None):
            marginal_gains = []
            for u in tqdm(cim.inactive_nodes(sim_graph), desc='Nodes examined', leave=None):
                agents_copy[self.curr_agent_id].seed = agents_copy[self.curr_agent_id].seed + [u]
                marginal_gain = cim.simulation(sim_graph, self.diff_model, agents=agents_copy, r=self.r)[agents_copy[self.curr_agent_id].name] - agents_copy[self.curr_agent_id].spread
                marginal_gains.append((u, marginal_gain))
                agents_copy[self.curr_agent_id].seed = agents_copy[self.curr_agent_id].seed[:-1]
            u, top_gain = max(marginal_gains, key=lambda x: x[1])
            # Update the agent's seed and spread
            agents_copy[self.curr_agent_id].seed.append(u)
            cim.activate_node(sim_graph, u, self.agents[self.curr_agent_id]) # Activate the top node
            agents_copy[self.curr_agent_id].spread += top_gain
        result_seed_set = agents_copy[self.curr_agent_id].seed[-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, agents_copy[self.curr_agent_id].spread