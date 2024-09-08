from src.single_agent.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
from tqdm import tqdm
from src.single_agent import influence_maximization as im

class CELF(SimulationBasedAlgorithm):
    """
    Paper: Leskovec et al. - "Cost-Effective Outbreak Detection in Networks"
    """

    name = 'celf'

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        sim_graph = self.graph.copy()
        seed = []
        marginal_gains = []
        # First iteration: Monte Carlo and store marginal gains
        for u in tqdm(im.inactive_nodes(sim_graph), desc="First Monte Carlo simulation"):
            tmp_seed = seed + [u]
            marginal_gains.append((u, im.simulation(sim_graph, self.diff_model, self.agent, tmp_seed, self.r)))
        marginal_gains = sorted(marginal_gains, key=lambda item: item[1], reverse=True)
        (top_node, spread) = marginal_gains.pop(0)
        seed.append(top_node)
        im.activate_node(sim_graph, top_node, self.agent)
        # Other iterations: use marginal gains to do an early stopping
        for _ in tqdm(range(self.budget-1), desc="Seed set"):
            check = False
            while not check:
                u = marginal_gains[0][0]
                tmp_seed = seed + [u]
                marginal_gain = im.simulation(sim_graph, self.diff_model, self.agent, tmp_seed, self.r) - spread
                marginal_gains[0] = (u, marginal_gain)
                marginal_gains = sorted(marginal_gains, key=lambda item: item[1], reverse=True)
                if marginal_gains[0][0] == u:
                    check = True
            (top_node, top_marginal_gain) = marginal_gains.pop(0)
            seed.append(top_node)
            spread += top_marginal_gain
        return seed
