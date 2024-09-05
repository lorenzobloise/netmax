from src.single_agent.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
from tqdm import tqdm
from src.single_agent import influence_maximization as im
from heapdict import heapdict

class CELF_PP(SimulationBasedAlgorithm):
    """
    Paper: Goyal et al. - "CELF++: Optimizing the Greedy Algorithm for Influence Maximization in Social Networks"
    """

    name = 'celfpp'

    class Node(object):

        def __init__(self, node):
            self.node = node
            self.mg1 = 0
            self.prev_best = None
            self.mg2 = 0
            self.mg2_already_computed = False
            self.flag = None
            self.list_index = 0

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        sim_graph = self.graph.copy()
        seed_set = []
        Q = heapdict() # Priority Queue based on marginal gain 1
        last_seed = None
        cur_best = None
        node_data_list = []
        # First iteration: Monte Carlo and store marginal gains
        for node in tqdm(sim_graph.nodes, desc="First Monte Carlo simulation"):
            node_data = CELF_PP.Node(node)
            node_data.mg1 = im.simulation(sim_graph, self.diff_model, self.agent, [node_data.node], self.r)
            node_data.prev_best = cur_best
            node_data.flag = 0
            cur_best = cur_best if cur_best and cur_best.mg1 > node_data.mg1 else node_data
            node_data_list.append(node_data)
            node_data.list_index = len(node_data_list) - 1
            Q[node_data.list_index] = -node_data.mg1
        # Other iterations: use marginal gains to do an early stopping
        progress_bar = tqdm(total=self.budget, desc="Seed Set", unit="it")
        while len(seed_set) < self.budget:
            node_idx, _ = Q.peekitem()
            node_data = node_data_list[node_idx]
            if not node_data.mg2_already_computed:
                node_data.mg2 = im.simulation(sim_graph, self.diff_model, self.agent, [node_data.node] + [cur_best.node], self.r)
                node_data.mg2_already_computed = True
            if node_data.flag == len(seed_set):
                seed_set.append(node_data.node)
                progress_bar.update(1)
                del Q[node_idx]
                last_seed = node_data
                continue
            elif node_data.prev_best == last_seed:
                node_data.mg1 = node_data.mg2
            else:
                node_data.mg1 = im.simulation_delta(sim_graph, self.diff_model, self.agent, seed_set+[node_data.node], seed_set, self.r)
                node_data.prev_best = cur_best
                node_data.mg2 = im.simulation_delta(sim_graph, self.diff_model, self.agent, seed_set+[cur_best.node]+[node_data.node], seed_set+[cur_best.node], self.r)
            node_data.flag = len(seed_set)
            cur_best = cur_best if cur_best and cur_best.mg1 > node_data.mg1 else node_data
            Q[node_idx] = -node_data.mg1
        return seed_set