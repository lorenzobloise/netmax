from src.multi_agent.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm

class HighestOutDegree(ProxyBasedAlgorithm):
    """
    This algorithm selects the nodes with the highest out-degree in the network.
    """
    name = 'outdeg'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)

    def run(self):
        # TODO
        pass