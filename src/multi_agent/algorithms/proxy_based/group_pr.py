from src.multi_agent.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm

class Group_PR(ProxyBasedAlgorithm):
    """
    Paper: Liu et al. - "Influence Maximization over Large-Scale Social Networks A Bounded Linear Approach"

    """
    name = 'group_pr'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)

    def run(self):
        # TODO
        pass