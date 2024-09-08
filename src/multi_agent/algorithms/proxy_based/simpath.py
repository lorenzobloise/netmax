from src.multi_agent.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm

class SimPath(ProxyBasedAlgorithm):
    """
    Can only be used under Linear Threshold diffusion model.
    Paper: Goyal et al. - "SimPath: An Efficient Algorithm for
    """
    name = 'simpath'

    class Node(object):

        def __init__(self, node, spd=0, spd_induced=0, flag=False):
            self.node = node
            self.spd = 0
            self.spd_induced = 0
            self.flag = flag

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)
        if diff_model.name != 'lt':
            raise ValueError(f"SimPath can only be executed under Linear Threshold diffusion model (set the argument at 'lt')")
        self.lookahead = 5
        self.eta = 0.001


    def run(self):
        # TODO
        pass