import math
import random
from netmax.algorithms.sketch_based.sketch_based import SketchBasedAlgorithm
import networkx as nx
from netmax.agent import Agent
import copy
import numpy as np
import logging
from tqdm import tqdm

class TIMp(SketchBasedAlgorithm):
    """
    With respect to TIM, TIM+ adds another phase between the KPT estimation and the node selection ones, which is the
    KPT refinement: in fact, KPT can be considerably smaller and having this phase can both:
    1) Significantly reduce the number of generated RR sets, so better computation time
    2) Improve the accuracy of the result
    """

    name = 'tim_p'

    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.sum_of_budgets = np.sum([a.budget for a in self.agents])  # For multi-agent setting, we have to take the sum of the budgets
        self.n = len(self.graph.nodes) - 1 # Number of nodes minus one
        self.m = len(self.graph.edges) # Number of edges
        self.theta = None # Number of RR sets
        self.kpt = None  # KPT parameter is crucial for determining the optimal number of RR sets to generate
        self.rr_sets = None # List that contains the RR sets
        self.occurrences = None # Dictionary <node: list> where the list contains the indexes of the RR sets where the node occurs
        self.l = 1 # Constant used for computing epsilon_prime and lambda
        self.epsilon = 0.2 # Constant used for computing epsilon_prime
        # Constant used for computing lambda
        self.epsilon_prime = 5 * math.pow((self.l * (self.epsilon**2))/(self.sum_of_budgets + self.l), 1/3)
        # Used along with KPT for computing optimal number of RR sets
        self._lambda = (2 + self.epsilon_prime) * self.l * self.n * math.log(self.n) * math.pow(self.epsilon_prime, -2)
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
        First phase of the TIM+ algorithm: estimate KPT, which is used to determine the optimal number of RR sets.
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
            for rr_set in self.rr_sets:  # For every generated RR sets
                in_degree_sum=0
                for node in rr_set:
                    # Sum the in degree of the nodes (the superclass method is called to handle the signed network case)
                    in_degree_sum += self.__in_degree_positive_edges__(node)
                kappa = 1 - (1 - (in_degree_sum / self.m))**self.sum_of_budgets
                sum += kappa
            if (sum/c_i) > (1/(2**i)): # The termination criterion is met, thus we end the parameter estimation
                progress_bar.update(progress_bar.total - i + 1)
                return self.n * (sum / (2 * c_i))
            progress_bar.update(1)
        progress_bar.close()
        # If the termination criteria is never met, we return 1 as a neutral value for KPT
        return 1

    def __fraction_of_covered_rr_sets__(self, s_k):
        """
        Returns the fraction of RR sets covered by the set containing the nodes occurring in most seed sets.
        :param s_k: the input set.
        :return: the fraction of RR sets covered by s_k.
        """
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
        """
        Intermediate phase of the TIM+ algorithm: use a part of the RR sets to return a new KPT value, which can be
        significantly smaller than the first one.
        :return: the new KPT value.
        """
        self.theta = math.floor(self._lambda / self.kpt) # Number of RR sets to generate
        s_k = [] # List of the top nodes
        occurrences = {v: set() for v in self.graph.nodes} # Initialize the occurrences dictionary
        for i in range(len(self.rr_sets)):
            for node in self.rr_sets[i]:
                occurrences[node].add(i)
        for _ in range(self.sum_of_budgets):
            # Pick node that covers the most reverse reachable sets and add it into s_k
            top_node = max(occurrences.items(), key=lambda x: len(x[1]))[0]
            s_k.append(top_node)
            # Remove all RR sets that are covered by this node
            self.rr_sets = [rr_set for idx, rr_set in enumerate(self.rr_sets) if idx not in occurrences[top_node]]
            # Update also the occurrences dictionary removing the indexes of the removed RR sets
            occurrences = {v: (occurrences[v].difference(occurrences[top_node])) for v in self.graph.nodes if v not in s_k}
        progress_bar = tqdm(total=max(1,self.theta-len(self.rr_sets)), desc="KPT refinement")
        # If there are more RR sets than the ones needed
        if self.theta <= len(self.rr_sets):
            # Take the first theta RR sets
            self.rr_sets = self.rr_sets[self.theta:]
            progress_bar.update(1)
        else:
            # Add the remaining RR sets
            rr_sets_tmp = self.rr_sets.copy()
            self.__build_reverse_reachable_sets__(num=self.theta-len(self.rr_sets), progress_bar=progress_bar)
            self.rr_sets.extend(rr_sets_tmp)
        # Get the fraction of the RR sets that the top nodes cover
        f = self.__fraction_of_covered_rr_sets__(s_k)
        # And compute the new KPT value accordingly
        kpt_prime = f * (self.n / (1 + self.epsilon_prime))
        # Return the most accurate lower bound of the two KPT values
        return max(self.kpt, kpt_prime)

    def __node_selection__(self, agents):
        """
        Picks the node that covers the most reverse reachable sets.
        :param agents_copy: The deep copy of the 'agents' dictionary.
        """
        top_node = max(self.occurrences.items(), key=lambda x: len(x[1]))[0] # Pick the node that covers the most RR sets
        agents[self.curr_agent_id].seed.append(top_node) # Add it into the seed set
        # Remove all reverse reachable sets that are covered by the node
        self.rr_sets = [rr_set for idx, rr_set in enumerate(self.rr_sets) if idx not in self.occurrences[top_node]]
        # Update also the occurrences dictionary removing the indexes of the removed RR sets
        self.occurrences = {v: self.occurrences[v].difference(self.occurrences[top_node]) for v in
                            self.graph.nodes if not self.__in_some_seed_set__(v, agents)}

    def run(self):
        """
        :return: The nodes to add in the seed set of the current agent and the spreads for each agent.
        """
        if self.kpt is None:
            self.kpt = self.__kpt_estimation__() # First phase
            self.kpt = self.__kpt_refinement__() # Intermediate phase
            self.theta = math.floor(self._lambda / self.kpt) # Number of RR sets to generate
            self.occurrences = {v: set() for v in self.graph.nodes} # Initialize the occurrences dictionary
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