from multi_agent.algorithms.sketch_based.sketch_based import SketchBasedAlgorithm
import networkx as nx
from multi_agent.agent import Agent
import random
import copy
from tqdm import tqdm
import numpy as np
import math

class RIS(SketchBasedAlgorithm):
    """
    Paper: Borgs et al. - "Maximizing Social Influence in Nearly Optimal Time" (2014)
    """

    name = 'ris'

    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.sum_of_budgets = np.sum([a.budget for a in self.agents])  # For multi-agent setting, we have to take the sum of the budgets
        self.n = len(self.graph.nodes) - 1
        self.m = len(self.graph.edges)
        self.rr_sets = None
        self.occurrences = None
        self.epsilon = 0.2
        self.alpha = 2
        self.l = 1
        self.tau = self.sum_of_budgets * (self.n + self.m) * math.log(self.n / math.pow(self.epsilon, 3)) + self.alpha * self.l

    def __build_reverse_reachable_sets__(self):
        self.rr_sets = []
        for _ in tqdm(range(math.floor(self.tau)), desc="Building Random Reverse Reachable Sets"):
            random_node = random.choice(list(self.graph.nodes))
            self.rr_sets.append(self.__generate_random_reverse_reachable_set__(random_node))
        self.occurrences = {v: [] for v in self.graph.nodes}
        # Pick node that covers the most reverse reachable sets
        for i in range(len(self.rr_sets)):
            for node in self.rr_sets[i]:
                self.occurrences[node].append(i)

    def __in_some_seed_set__(self, v, agents):
        for a in agents:
            if v in a.seed:
                return True
        return False

    def __node_selection__(self, agents):
        top_node = max(self.occurrences.items(), key=lambda x: len(x[1]))[0]
        # Add it into the seed set
        agents[self.curr_agent_id].seed.append(top_node)
        # Remove all reverse reachable sets that are covered by the node
        self.rr_sets = [rr_set for idx, rr_set in enumerate(self.rr_sets) if idx not in self.occurrences[top_node]]
        self.occurrences = {v: [idx for idx in self.occurrences[v] if idx not in self.occurrences[top_node]] for v in
                            self.graph.nodes if not self.__in_some_seed_set__(v, agents)}

    def run(self):
        # Generate random reverse reachable sets
        if self.rr_sets is None:
            self.__build_reverse_reachable_sets__()
        agents_copy = copy.deepcopy(self.agents)
        for _ in range(self.budget):
            self.__node_selection__(agents_copy)
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: 0 for a in agents_copy}