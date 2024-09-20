from multi_agent.algorithms.sketch_based.sketch_based import SketchBasedAlgorithm
import networkx as nx
from multi_agent.agent import Agent
import random
import copy
from tqdm import tqdm

class RIS(SketchBasedAlgorithm):
    """
    Paper: Borgs et al. - "Maximizing Social Influence in Nearly Optimal Time" (2014)

    """
    name = 'ris'

    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.rr_sets = None
        self.occurrences = None

    def __build_reverse_reachable_sets__(self):
        self.rr_sets = []
        for _ in tqdm(range(self.r), desc="Building Random Reverse Reachable Sets"):
            sketch = self.__generate_sketch__()
            random_node = random.choice(list(self.graph.nodes))
            self.rr_sets.append(nx.ancestors(sketch, random_node))
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

    def run(self):
        # Generate random reverse reachable sets
        if self.rr_sets is None:
            self.__build_reverse_reachable_sets__()
        agents_copy = copy.deepcopy(self.agents)
        for _ in range(self.budget):
            top_node = max(self.occurrences.items(), key=lambda x: len(x[1]))[0]
            # Add it into the seed set
            agents_copy[self.curr_agent_id].seed.append(top_node)
            # Remove all reverse reachable sets that are covered by the node
            self.rr_sets = [rr_set for idx, rr_set in enumerate(self.rr_sets) if idx not in self.occurrences[top_node]]
            self.occurrences = {v: [idx for idx in self.occurrences[v] if idx not in self.occurrences[top_node]] for v in self.graph.nodes if not self.__in_some_seed_set__(v, agents_copy)}
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: 0 for a in agents_copy}