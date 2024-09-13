import copy
from multi_agent.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
import multi_agent.competitive_influence_maximization as cim
from tqdm import tqdm
import logging

class MCGreedy(SimulationBasedAlgorithm):

    name = 'mcgreedy'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def __update_spreads__(self, agents, spreads):
        for agent in agents:
            if agent.name in spreads:
                agent.spread = spreads[agent.name]
            else:
                agent.spread = 0

    def run(self):
        self.logger.info(f"Agent {self.agents[self.curr_agent_id].name} (id: {self.curr_agent_id}) started MCGreedy")
        agents_copy = copy.deepcopy(self.agents)
        for _ in range(self.budget):
            marginal_gains = []
            last_spread = agents_copy[self.curr_agent_id].spread
            for u in tqdm(cim.inactive_nodes(self.graph), desc='Nodes examined', leave=None):
                agents_copy[self.curr_agent_id].seed = agents_copy[self.curr_agent_id].seed + [u]
                spreads = cim.simulation(self.graph, self.diff_model, agents=agents_copy, r=self.r)
                marginal_gain = spreads[self.agents[self.curr_agent_id].name] - last_spread
                marginal_gains.append((u, marginal_gain, spreads))
                agents_copy[self.curr_agent_id].seed = agents_copy[self.curr_agent_id].seed[:-1]
            u, top_gain, spreads = max(marginal_gains, key=lambda x: x[1])
            self.__update_spreads__(agents_copy, spreads)
            self.logger.info(f"Marginal gains: {sorted(marginal_gains, key=lambda x: x[1])}")
            # Update the agent's seed and spread
            agents_copy[self.curr_agent_id].seed.append(u)
            cim.activate_node(self.graph, u, self.agents[self.curr_agent_id]) # Activate the top node
            self.logger.info(f"Agent {self.agents[self.curr_agent_id].name} (id: {self.curr_agent_id}) has chosen node {u}")
            self.logger.info(f"Spread before: {last_spread}, top_gain: {top_gain}")
            #agents_copy[self.curr_agent_id].spread += top_gain
            self.logger.info(f"Spread after: {agents_copy[self.curr_agent_id].spread}")
        self.logger.info(f"Agent {self.agents[self.curr_agent_id].name} (id: {self.curr_agent_id}) finished MCGreedy")
        result_seed_set = agents_copy[self.curr_agent_id].seed[-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: a.spread for a in agents_copy}