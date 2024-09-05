from src.single_agent.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
from tqdm import tqdm
from src.single_agent import influence_maximization as im
import networkx as nx

class CGA(SimulationBasedAlgorithm):
    """
    Paper: Wang et al. - "Community-based Greedy Algorithm for Mining Top-K Influential Nodes in Mobile Social Networks"
    """

    name = 'cga'

    def __init__(self, graph, agent, budget, diff_model, r=10000):
        super().__init__(graph, agent, budget, diff_model, r)

    def run(self):
        sim_graph = self.graph.copy()
        seed_set = []
        # Detect communities
        C = nx.community.louvain_communities(sim_graph, weight='p')
        M = len(C)
        seed_communities = {i: [] for i in range(M)}
        R = {}
        s = {}
        for k in range(1,self.budget+1):
            R[(0,k)] = 0
            s[(0,k)] = 0
        for m in range(1,M+1):
            R[(m,0)] = 0
        for k in tqdm(range(1,self.budget+1), desc="Seed set"):
            for m in tqdm(range(1,M+1), desc="Communities"):
                spreads_m = [im.simulation_delta(sim_graph, self.diff_model, self.agent, seed_set+[v], seed_set, self.r, C[m-1]) for v in C[m-1]]
                # Take maximum marginal gain for the community m
                delta_R_m = max(spreads_m)
                R[(m,k)] = max(R[(m-1,k)],R[(M,k-1)]+delta_R_m)
                if R[(m-1,k)] >= R[(M,k-1)] + delta_R_m:
                    s[(m,k)] = s[(m-1,k)]
                else:
                    s[(m,k)] = m
            j = s[(M,k)]
            marg_gains = {v_i: im.simulation(sim_graph, self.diff_model, self.agent, seed_communities[j-1]+[v_i], self.r, C[j-1]) - im.simulation(sim_graph, self.diff_model, self.agent, seed_communities[j-1], self.r) for v_i in C[j-1]}
            v_max = max(marg_gains, key=marg_gains.get)
            seed_communities[j-1].append(v_max)
            seed_set.append(v_max)
        return seed_set