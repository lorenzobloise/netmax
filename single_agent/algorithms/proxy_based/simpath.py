from heapdict import heapdict
from single_agent.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm

class SimPath(ProxyBasedAlgorithm):
    """
    Can only be used under Linear Threshold diffusion model.
    Paper: Goyal et al. - "SimPath: An Efficient Algorithm for Influence Maximization under the Linear Threshold Model"
    """

    name = 'simpath'

    class Node(object):

        def __init__(self, node, spd=0, spd_induced=0, flag=False):
            self.node = node
            self.spd = 0
            self.spd_induced = 0
            self.flag = flag

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)
        if diff_model.name != 'lt':
            raise ValueError(f"SimPath can only be executed under Linear Threshold diffusion model (set the argument at 'lt')")
        self.lookahead = 5
        self.eta = 0.001

    def __simpath_spread__(self, seed, U):
        spd, spd_induced = 0, 0
        for u in seed:
            if self.graph.out_degree(u) > 0:
                spd_new, spd_induced_new = self.__backtrack__(u, (set(self.graph.nodes) - set(seed)).union({u}), U)
                spd += spd_new
                spd_induced += spd_induced_new
        return spd, spd_induced

    def __backtrack__(self, u, W, U):
        Q = [u]
        spd, spd_induced, pp, D = 1, 1, 1, {}
        while len(Q) > 0:
            Q, D, spd, spd_induced, pp = self.__forward__(Q, D, spd, spd_induced, pp, W, U)
            u = Q.pop()
            del D[u]
            if len(Q) == 0:
                break
            v = Q[-1]
            pp = pp / self.graph.edges[v, u]['p']
        return spd, spd_induced

    def __exists_y__(self, x, Q, D, W):
        """
        As specified in the paper, this function returns a node y that is in the out-neighborhood of x
        and not in Q and not in D[x] but it is in W
        """
        out_neighbors = {v for (x, v) in self.graph.out_edges(x)}
        for y in out_neighbors:
            if y not in Q and y not in D[x] and y in W:
                return True, y
        return False, None

    def __forward__(self, Q, D, spd, spd_induced, pp, W, U):
        x = Q[-1]
        if x not in list(D.keys()):
            D[x] = []
        exists_y, y = self.__exists_y__(x, Q, D, W)
        while exists_y:
            if pp * self.graph.edges[x, y]['p'] < self.eta:
                if x not in list(D.keys()):
                    D[x] = []
                D[x].append(y)
            else:
                Q.append(y)
                pp = pp * self.graph.edges[x, y]['p']
                spd += pp
                if x not in list(D.keys()):
                    D[x] = []
                D[x].append(y)
                x = Q[-1]
                if x not in list(D.keys()):
                    D[x] = []
                for v in U:
                    if v not in Q:
                        spd_induced += pp
            exists_y, y = self.__exists_y__(x, Q, D, W)
        return Q, D, spd, spd_induced, pp

    def __get_node_from_queue__(self, queue, node):
        for u in queue:
            if u.node == node:
                return u
        return None

    def run(self):
        import networkx.algorithms.approximation.vertex_cover as vc
        # Compute the vertex cover of input graph
        C = vc.min_weighted_vertex_cover(self.graph.to_undirected(as_view=True))
        queue = heapdict()  # Priority queue based on the spread value
        for u in C:
            in_neighbor = {v for (v, u) in self.graph.in_edges(u)}
            U = set((set(self.graph.nodes) - C).intersection(in_neighbor))
            spd, spd_induced = self.__simpath_spread__([u], U)
            node_data = self.Node(u, spd, spd_induced)
            # Add node to the queue
            queue[node_data] = -node_data.spd
        for v in set(self.graph.nodes) - C:
            # Compute the spread value of v as specified by Theorem 2
            spread_v = 1 + sum([self.graph.edges[v,u]['p'] * self.__get_node_from_queue__(queue, u).spd_induced for (v, u) in self.graph.out_edges(v)])
            queue[self.Node(v, spread_v)] = -spread_v
        seed_set = []
        spread = 0
        while len(seed_set) < self.budget:
            U = []
            if len(queue) < self.lookahead:
                self.lookahead = len(queue)
            for _ in range(self.lookahead):
                curr_node_data, _ = queue.popitem()
                U.append(curr_node_data.node)
                queue[curr_node_data] = -curr_node_data.spd
            _, spd_induced_seed = self.__simpath_spread__(seed_set, U)
            for x in U:
                node_data_x = self.__get_node_from_queue__(queue, x)
                if node_data_x.flag:
                    seed_set.append(x)
                    spread += node_data_x.spd
                    del queue[node_data_x]
                    break
                _, spd_induced_seed_x = self.__backtrack__(x, set(self.graph.nodes)-set(seed_set), [])
                spread_seed_x = spd_induced_seed_x + spd_induced_seed
                marginal_gain_x = spread_seed_x - spread
                node_data_x.flag = True
                node_data_x.spd = marginal_gain_x
                queue[node_data_x] = -node_data_x.spd
        return seed_set