from src.single_agent.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
from tqdm import tqdm
from src.single_agent import influence_maximization as im

class MCGreedy(SimulationBasedAlgorithm):

    name = 'mcgreedy'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        sim_graph = self.graph.copy()
        seed = []
        spread = 0
        for _ in tqdm(range(self.budget), desc="Seed set"):
            marginal_gains = []
            for u in tqdm(im.inactive_nodes(sim_graph), desc="Inactive nodes"):
                tmp_seed = seed + [u]
                marginal_gain = im.simulation(sim_graph, self.diff_model, self.agent, tmp_seed, self.r) - spread
                marginal_gains.append((u,marginal_gain))
            marginal_gains = sorted(marginal_gains, key=lambda item: item[1], reverse=True)
            (top_node, top_marginal_gain) = marginal_gains.pop(0)
            seed.append(top_node)
            im.activate_node(sim_graph, top_node, self.agent)
            spread += top_marginal_gain
        return seed