from tqdm import tqdm
from netmax.algorithms.sketch_based.sketch_based import SketchBasedAlgorithm
import networkx as nx
from netmax.agent import Agent
import copy

class StaticGreedy(SketchBasedAlgorithm):
    """
    Paper: Chen et al. - "StaticGreedy: Solving the Scalability-Accuracy Dilemma in Influence Maximization" (2013).
    This method produces a number of Monte Carlo snapshots at the beginning, and uses this same set of snapshots
    (thus, static) in all iterations, instead of producing a huge number of Monte Carlo simulations in every iteration.
    """

    name = 'static_greedy'

    class Snapshot(object):
        """
        Class that encapsulates useful information on the snapshots.
        """

        _idx = 0 # Static variable of the class Snapshot

        def __init__(self, sketch, reached_nodes, reached_from_nodes):
            self.id = StaticGreedy.Snapshot._idx
            StaticGreedy.Snapshot._idx += 1
            self.sketch = sketch # Subgraph induced by an instance of the influence process
            # Dictionary which contains, for each node u, the set of nodes that can be reached from u
            self.reached_nodes = reached_nodes
            # Dictionary which contains, for each node u, the set of nodes from which u can be reached
            self.reached_from_nodes = reached_from_nodes

    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.snapshots = None # List containing the snapshots
        self.marginal_gains = {} # Dictionary of marginal gains for each node
        self.R = 100 # Number of snapshots

    def __generate_single_snapshot__(self):
        """
        Generates a snapshot.
        :return: A snapshot, along with the dictionary <node: reached_nodes> and
        the dictionary <node: reached_from_nodes>
        """
        # 1) Sample each edge (u,v) from the graph according to its probability p(u,v)
        sketch = self.__generate_sketch__()
        # 2) For each node u, compute:
        # 2.1) The reached nodes R(G_i, u)
        reached_nodes = {u: list(nx.descendants(sketch, u)) for u in sketch.nodes}
        # 2.2) The nodes from which u is reached U(G_i, u)
        reached_from_nodes = {u: list(nx.ancestors(sketch, u)) for u in sketch.nodes}
        return sketch, reached_nodes, reached_from_nodes

    def __produce_snapshots__(self):
        """
        Generates R snapshots and stores them.
        """
        self.snapshots = []
        for _ in tqdm(range(self.R), desc="Creating snapshots"):
            # Generate a snapshot and encapsulate it inside the Snapshot class
            sketch, reached_nodes, reached_from_nodes = self.__generate_single_snapshot__()
            snapshot = StaticGreedy.Snapshot(sketch, reached_nodes, reached_from_nodes)
            self.snapshots.append(snapshot)
            # Update the marginal gain dictionary
            for v in self.graph.nodes:
                self.marginal_gains[v] = self.marginal_gains.get(v, 0) + len(snapshot.reached_nodes[v])

    def __take_best_node__(self):
        """
        Takes the node with the highest marginal gain.
        :return: the best node and its marginal gain.
        """
        self.marginal_gains = dict(sorted(self.marginal_gains.items(), key=lambda x: x[1])) # Sort the dictionary
        best_node, marg_gain = self.marginal_gains.popitem() # Pop the first element
        return best_node, marg_gain

    def __discount_marginal_gains__(self, v):
        """
        When a node v is selected as seed, directly discount the marginal gain of other nodes by the marginal gain
        shared by these nodes and v.
        :param v: the node we're considering.
        """
        for snapshot in self.snapshots:
            for w in snapshot.reached_nodes[v]: # For each node w reached from v
                for u in snapshot.reached_from_nodes[w]: # For each node u that can reach u
                    if u != v: # If u is not the node we started from (v)
                        snapshot.reached_nodes[u].remove(w) # Remove w from the nodes u can reach
                        self.marginal_gains[u] -= 1 # Discount the marginal gain of u

    def run(self):
        """
        :return: The nodes to add in the seed set of the current agent and the spreads for each agent.
        """
        # Generate the Monte Carlo snapshots if it's the first turn of the first round
        if self.snapshots is None:
            self.__produce_snapshots__()
        nodes_added = 0
        agents_copy = copy.deepcopy(self.agents)
        # Repeats until the budget is fulfilled (in the InfluenceMaximization class, inside the run method, the algorithm
        # is always invoked with the budget parameter set to 1, but we preferred to write the code in a more general way)
        while nodes_added < self.budget:
            v_max, marg_gain = self.__take_best_node__() # Greedy selection
            agents_copy[self.curr_agent_id].seed.append(v_max) # Add the best node to the seed set of the current agent
            agents_copy[self.curr_agent_id].spread += marg_gain # Update the spread of the current agent
            nodes_added += 1
            self.__discount_marginal_gains__(v_max) # Discount the marginal gains of the nodes
        # Return the new nodes to add to the seed set and the spread
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: a.spread for a in agents_copy}