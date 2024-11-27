from netmax.algorithms.sketch_based.sketch_based import SketchBasedAlgorithm
import networkx as nx
from netmax.agent import Agent
import random
import copy
from tqdm import tqdm
import numpy as np
import math

class RIS(SketchBasedAlgorithm):
    """
    Paper: Borgs et al. - "Maximizing Social Influence in Nearly Optimal Time" (2014).
    In RIS, the influence of any seed set is estimated by selecting random nodes and seeing the portion of the
    randomly selected nodes which can be reached by S, called Reverse Reachable sets (RR sets).
    """

    name = 'ris'

    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.sum_of_budgets = np.sum([a.budget for a in self.agents])  # For multi-agent setting, we have to take the sum of the budgets
        self.n = len(self.graph.nodes) - 1 # Number of nodes minus one
        self.m = len(self.graph.edges) # Number of edges
        self.rr_sets = None # List that contains the RR sets
        self.occurrences = None # Dictionary <node: list> where the list contains the indexes of the RR sets where the node occurs
        self.epsilon = 0.2 # Constant used to compute the number of RR sets
        # Number of RR sets
        self.tau = self.sum_of_budgets * (self.n + self.m) * math.log(self.n / math.pow(self.epsilon, 3))

    def __build_reverse_reachable_sets__(self):
        """
        Builds the list containing the reverse reachable sets.
        """
        self.rr_sets = []
        for _ in tqdm(range(math.floor(self.tau)), desc="Building Random Reverse Reachable Sets"):
            random_node = random.choice(list(self.graph.nodes)) # Select a random node
            self.rr_sets.append(self.__generate_random_reverse_reachable_set__(random_node)) # Invoke the superclass method
        self.occurrences = {v: [] for v in self.graph.nodes} # Build the nodes' occurrences
        for i in range(len(self.rr_sets)):
            for node in self.rr_sets[i]:
                self.occurrences[node].append(i)

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
        self.occurrences = {v: [idx for idx in self.occurrences[v] if idx not in self.occurrences[top_node]] for v in
                            self.graph.nodes if not self.__in_some_seed_set__(v, agents_copy)}

    def run(self):
        """
        :return: The nodes to add in the seed set of the current agent and the spreads for each agent.
        """
        # Generate the random reverse reachable sets if it's the first turn of the first round
        if self.rr_sets is None:
            self.__build_reverse_reachable_sets__()
        agents_copy = copy.deepcopy(self.agents)
        # Repeats until the budget is fulfilled (in the InfluenceMaximization class, inside the run method, the algorithm
        # is always invoked with the budget parameter set to 1, but we preferred to write the code in a more general way)
        for _ in range(self.budget):
            self.__node_selection__(agents_copy)
        # Return the new nodes to add to the seed set and the spread
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: 0 for a in agents_copy}