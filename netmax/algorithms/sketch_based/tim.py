import math
import random
from netmax.algorithms.sketch_based.sketch_based import SketchBasedAlgorithm
import networkx as nx
from netmax.agent import Agent
import copy
import numpy as np
import logging
from tqdm import tqdm

class TIM(SketchBasedAlgorithm):
    """
    Paper: Tang et al. - "Influence Maximization: Near Optimal Time Complexity Meets Practical Efficiency" (2014).
    TIM reduces the number of RR sets required to ensure the same theoretical bound as RIS, by doing two distinct
    phases: KPT estimation (where KPT is used for determining the optimal number of RR sets) and node selection.
    """

    name = 'tim'

    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.sum_of_budgets = np.sum([a.budget for a in self.agents])  # For multi-agent setting, we have to take the sum of the budgets
        self.n = len(self.graph.nodes) - 1 # Number of nodes minus one
        self.m = len(self.graph.edges) # Number of edges
        self.kpt = None # KPT parameter is crucial for determining the optimal number of RR sets to generate
        self.rr_sets = None # List that contains the RR sets
        self.occurrences = None # Dictionary <node: list> where the list contains the indexes of the RR sets where the node occurs
        self.l = 1 # Constant used for computing lambda
        self.epsilon = 0.2 # Constant used for computing lambda
        # Used along with KPT for computing optimal number of RR sets
        self._lambda = (8 + 2 * self.epsilon) * self.n * (self.l * math.log(self.n) + math.log(__binomial_coefficient__(self.n, self.sum_of_budgets)) + math.log(2)) * math.pow(self.epsilon, -2)
        self.theta = None # Number of RR sets
        self.logger = logging.getLogger()

    def __build_reverse_reachable_sets__(self, num, progress_bar=None):
        """
        Builds the list containing the reverse reachable sets.
        :param num: the number of RR sets to generate.
        :param progress_bar: optional progress bar to make the process visible.
        """
        self.rr_sets = []
        for _ in range(num):
            # Invoke the superclass method
            rr_set = self.__generate_random_reverse_reachable_set__(random.choice(list(self.graph.nodes)))
            self.rr_sets.append(rr_set)
            if progress_bar is not None:
                progress_bar.update(1)

    def __kpt_estimation__(self):
        """
        First phase of the TIM algorithm: estimate KPT, which is used to determine the optimal number of RR sets.
        KPT is estimated by building some of the total RR sets.
        :return: KPT value.
        """
        progress_bar = tqdm(total=math.floor(math.log(self.n,2))-1, desc="KPT estimation")
        # Number of iterations that depends on the size of the network
        for i in range(1, math.floor(math.log(self.n,2))):
            # Coefficient that indicates how many RR sets to generate in the current iteration
            c_i = math.floor((6 * self.l * math.log(self.n) + 6 * math.log(math.log(self.n,2)))*(2**i))
            sum = 0
            self.__build_reverse_reachable_sets__(num=c_i)
            for rr_set in self.rr_sets: # For every generated RR sets
                in_degree_sum = 0
                for node in rr_set:
                    # Sum the in degree of the nodes (the superclass method is called to handle the signed network case)
                    in_degree_sum += self.__in_degree_positive_edges__(node)
                kappa = 1 - (1 - (in_degree_sum / self.m))**self.sum_of_budgets
                sum += kappa
            if (sum/c_i) > (1/(2**i)): # The termination criterion is met, thus we end the parameter estimation
                progress_bar.update(progress_bar.total - i + 1)
                return self.n * sum / (2 * c_i)
            progress_bar.update(1)
        progress_bar.close()
        # If the termination criteria is never met, we return 1 as a neutral value for KPT
        return 1

    def __node_selection__(self, agents_copy):
        """
        Picks the node that covers the most reverse reachable sets.
        :param agents_copy: The deep copy of the 'agents' dictionary.
        """
        top_node = max(self.occurrences.items(), key=lambda x: len(x[1]))[0] # Pick the node that covers the most RR sets
        agents_copy[self.curr_agent_id].seed.append(top_node) # Add it into the seed set
        # Remove all reverse reachable sets that are covered by the node
        self.rr_sets = [rr_set for idx, rr_set in enumerate(self.rr_sets) if idx not in self.occurrences[top_node]]
        # Update also the occurrences dictionary removing the indexes of the removed RR sets
        self.occurrences = {v: self.occurrences[v].difference(self.occurrences[top_node]) for v in
                            self.graph.nodes if not self.__in_some_seed_set__(v, agents_copy)}

    def run(self):
        """
        :return: The nodes to add in the seed set of the current agent and the spreads for each agent.
        """
        if self.kpt is None: # If it's the first turn of the first round, estimate KPT and build RR sets
            self.kpt = self.__kpt_estimation__()
            self.theta = math.floor(self._lambda / self.kpt) # Number of RR sets to generate
            progress_bar = tqdm(total=self.theta, desc="Building Reverse Reachable Sets")
            self.__build_reverse_reachable_sets__(num=self.theta, progress_bar=progress_bar)
            self.occurrences = {v: set() for v in self.graph.nodes} # Initialize the occurrences
            for i in range(len(self.rr_sets)):
                for node in self.rr_sets[i]:
                    self.occurrences[node].add(i)
        agents_copy = copy.deepcopy(self.agents)
        # Repeats until the budget is fulfilled (in the InfluenceMaximization class, inside the run method, the algorithm
        # is always invoked with the budget parameter set to 1, but we preferred to write the code in a more general way)
        for _ in range(self.budget):
            self.__node_selection__(agents_copy)
        # Return the new nodes to add to the seed set and the spread
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: 0 for a in agents_copy}

# Utility function
def __binomial_coefficient__(n, k):
    """
    Efficient binomial coefficient computation, used in TIM algorithm.
    """
    C = [[-1 for _ in range(k+1)] for _ in range(n+1)]
    for i in range(n+1):
        for j in range(min(i, k+1)):
            # Base cases
            if j == 0 or j == i:
                C[i][j] = 1
            # Calculate value using previously stored values
            else:
                C[i][j] = C[i-1][j-1] + C[i-1][j]
    return C[n][k]