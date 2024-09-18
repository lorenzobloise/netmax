import random
from multi_agent.algorithms.sketch_based.sketch_based import SketchBasedAlgorithm
import networkx as nx
from multi_agent.agent import Agent
import copy

class StaticGreedy(SketchBasedAlgorithm):
    """
    Paper: Chen et al. - "StaticGreedy: Solving the Scalability-Accuracy Dilemma in Influence Maximization" (2013).
    This method produces a number of Monte Carlo snapshots at the beginning, and use this same set of snapshots
    (thus, static) in all iterations, instead of producing a huge number of Monte Carlo simulations in every iteration.
    """

    name = 'static_greedy'

    class Snapshot(object):

        _idx = 0

        def __init__(self, sketch, reached_nodes, reached_from_nodes):
            self.id = StaticGreedy.Snapshot._idx
            StaticGreedy.Snapshot._idx += 1
            self.sketch = sketch # Subgraph induced by an instance of the influence process
            self.reached_nodes = reached_nodes # Dictionary which contains, for each node u, the set of nodes that can be reached from u
            self.reached_from_nodes = reached_from_nodes # Dictionary which contains, for each node u, the set of nodes from which u can be reached


    def __init__(self, graph: nx.DiGraph, agents: list[Agent], curr_agent_id: int, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        self.snapshots = None # List which contains the snapshots
        self.marginal_gains = {a.id: {} for a in self.agents} # Dictionary of dictionaries (marginal gains) for each node and for each agent

    def __generate_snapshot__(self):
        # 1) Sample each edge (u,v) from the graph according to its probability p(u,v)
        sampled_edges = []
        for (u, v, attr) in self.graph.edges(data=True):
            r = random.random()
            if r < attr['p']:
                sampled_edges.append((u, v))
        sim_graph = nx.DiGraph()
        sim_graph.add_nodes_from([node for node in self.graph.nodes])
        sim_graph.add_edges_from(sampled_edges)
        # 2) For each node u, compute:
        # 2.1) The reached nodes R(G_i, u)
        reached_nodes = {u: list(nx.descendants(sim_graph, u)) for u in sim_graph.nodes}
        # 2.2) The nodes from which u is reached U(G_i, u)
        reached_from_nodes = {u: list(nx.ancestors(sim_graph, u)) for u in sim_graph.nodes}
        return sim_graph, reached_nodes, reached_from_nodes

    def __produce_snapshots__(self):
        self.snapshots = []
        for _ in range(self.r):
            sketch, reached_nodes, reached_from_nodes = self.__generate_snapshot__()
            snapshot = StaticGreedy.Snapshot(sketch, reached_nodes, reached_from_nodes)
            self.snapshots.append(snapshot)
            for v in self.graph.nodes:
                for a in self.agents:
                    if v not in self.marginal_gains[a.id].keys():
                        self.marginal_gains[a.id][v] = 0
                    self.marginal_gains[a.id][v] += len(snapshot.reached_nodes[v])

    def __take_best_node__(self):
        self.marginal_gains[self.curr_agent_id] = dict(sorted(self.marginal_gains[self.curr_agent_id].items(), key=lambda x: x[1]))
        best_node, marg_gain = self.marginal_gains[self.curr_agent_id].popitem()
        # Remove the best node from the other queues
        for a in self.agents:
            if best_node in self.marginal_gains[a.id].keys():
                del self.marginal_gains[a.id][best_node]
        return best_node, marg_gain

    def __discount_marginal_gains__(self, v):
        """
        When a node v is selected as seed, directly discount the marginal gain of other nodes by the marginal gain
        shared by these nodes and v.
        """
        for snapshot in self.snapshots:
            for w in snapshot.reached_from_nodes[v]:
                for u in snapshot.reached_nodes[v]:
                    if w in snapshot.reached_nodes[u]:
                        snapshot.reached_nodes[u].remove(w)
                    for a in self.agents:
                        self.marginal_gains[a.id][u] -= 1

    def run(self):
        # Generate the Monte Carlo snapshots
        if self.snapshots is None:
            self.__produce_snapshots__()
        # Greedy selection
        nodes_added = 0
        agents_copy = copy.deepcopy(self.agents)
        while nodes_added < self.budget:
            v_max, marg_gain = self.__take_best_node__()
            agents_copy[self.curr_agent_id].seed.append(v_max)
            agents_copy[self.curr_agent_id].spread += marg_gain
            nodes_added += 1
            self.__discount_marginal_gains__(v_max)
        result_seed_set = agents_copy[self.curr_agent_id].seed[:-self.budget] if self.budget > 1 else [agents_copy[self.curr_agent_id].seed[-1]]
        return result_seed_set, {a.name: a.spread for a in agents_copy}