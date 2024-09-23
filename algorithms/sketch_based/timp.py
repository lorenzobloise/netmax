import math
import random
from algorithms.sketch_based.sketch_based import SketchBasedAlgorithm
import networkx as nx
from agent import Agent
import copy
import numpy as np
import logging
from tqdm import tqdm

class TIMp(SketchBasedAlgorithm):

    name = 'tim_p'

    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.sum_of_budgets = np.sum([a.budget for a in self.agents])  # For multi-agent setting, we have to take the sum of the budgets
        self.n = len(self.graph.nodes) - 1
        self.m = len(self.graph.edges)
        self.theta = None
        self._lambda = None
        self.kpt = None
        self.rr_sets = None
        self.occurrences = None
        self.l = 1
        self.epsilon = 0.2
        self.epsilon_prime = 5 * math.pow((self.l * (self.epsilon**2))/(self.sum_of_budgets + self.l), 1/3)
        self._lambda = (2 + self.epsilon_prime) * self.l * self.n * math.log(self.n) * math.pow(self.epsilon_prime, -2)
        self.logger = logging.getLogger()

    def __in_some_seed_set__(self, v, agents):
        for a in agents:
            if v in a.seed:
                return True
        return False

    def __build_reverse_reachable_sets__(self, num, progress_bar=None):
        self.rr_sets = []
        for _ in range(num):
            rr_set = self.__generate_random_reverse_reachable_set__(random.choice(list(self.graph.nodes)))
            self.rr_sets.append(rr_set)
            if progress_bar is not None:
                progress_bar.update(1)

    def __kpt_estimation__(self):
        progress_bar = tqdm(total=math.floor(math.log(self.n,2))-1, desc="KPT estimation")
        for i in range(1, math.floor(math.log(self.n,2))):
            c_i = math.floor((6 * self.l * math.log(self.n) + 6 * math.log(math.log(self.n,2)))*(2**i))
            sum = 0
            self.__build_reverse_reachable_sets__(num=c_i)
            for rr_set in self.rr_sets:
                in_degree_sum=0
                for node in rr_set:
                    in_degree_sum += self.graph.in_degree(node)
                kappa = 1 - (1 - (in_degree_sum / self.m))**self.sum_of_budgets
                sum += kappa
            if (sum/c_i) > (1/(2**i)):
                progress_bar.update(progress_bar.total - i + 1)
                return self.n * (sum / (2 * c_i))
            progress_bar.update(1)
        progress_bar.close()
        return 1

    def __fraction_of_covered_rr_sets__(self, s_k):
        counter = 0
        for rr_set in self.rr_sets:
            cond = True
            for node in s_k:
                if node not in rr_set:
                    cond = False
                    break
            if cond:
                counter += 1
        return counter / len(self.rr_sets)

    def __kpt_refinement__(self):
        self.theta = math.floor(self._lambda / self.kpt)
        s_k = []
        occurrences = {v: set() for v in self.graph.nodes}
        for i in range(len(self.rr_sets)):
            for node in self.rr_sets[i]:
                occurrences[node].add(i)
        for _ in range(self.sum_of_budgets):
            # Pick node that covers the most reverse reachable sets and add it into s_k
            top_node = max(occurrences.items(), key=lambda x: len(x[1]))[0]
            s_k.append(top_node)
            # Remove all RR sets that are covered by this node
            self.rr_sets = [rr_set for idx, rr_set in enumerate(self.rr_sets) if idx not in occurrences[top_node]]
            occurrences = {v: (occurrences[v].difference(occurrences[top_node])) for v in self.graph.nodes if v not in s_k}
        progress_bar = tqdm(total=max(1,self.theta-len(self.rr_sets)), desc="KPT refinement")
        if self.theta <= len(self.rr_sets):
            # Take the first theta rr sets
            self.rr_sets = self.rr_sets[self.theta:]
            progress_bar.update(1)
        else:
            # Add the remaining rr sets
            rr_sets_tmp = self.rr_sets.copy()
            self.__build_reverse_reachable_sets__(num=self.theta-len(self.rr_sets), progress_bar=progress_bar)
            self.rr_sets.extend(rr_sets_tmp)
        f = self.__fraction_of_covered_rr_sets__(s_k)
        kpt_prime = f * (self.n / (1 + self.epsilon_prime))
        return max(self.kpt, kpt_prime)

    def __node_selection__(self, agents):
        top_node = max(self.occurrences.items(), key=lambda x: len(x[1]))[0]
        # Add it into the seed set
        agents[self.curr_agent_id].seed.append(top_node)
        # Remove all reverse reachable sets that are covered by the node
        self.rr_sets = [rr_set for idx, rr_set in enumerate(self.rr_sets) if idx not in self.occurrences[top_node]]
        self.occurrences = {v: self.occurrences[v].difference(self.occurrences[top_node]) for v in
                            self.graph.nodes if not self.__in_some_seed_set__(v, agents)}

    def run(self):
        if self.kpt is None:
            self.kpt = self.__kpt_estimation__()
            self.kpt = self.__kpt_refinement__()
            self.theta = math.floor(self._lambda / self.kpt)
            self.occurrences = {v: set() for v in self.graph.nodes}
            for i in range(len(self.rr_sets)):
                for node in self.rr_sets[i]:
                    self.occurrences[node].add(i)
        agents_copy = copy.deepcopy(self.agents)
        for _ in range(self.budget):
            self.__node_selection__(agents_copy)
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: 0 for a in agents_copy}