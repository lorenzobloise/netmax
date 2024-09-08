from src.multi_agent.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm

class UBound(ProxyBasedAlgorithm):
    """
    Paper: Zhou et al. - "UBLF: An Upper Bound Based Approach to Discover Influential Nodes in Social Networks"
    The UBound algorithm doesn't work if the graph has cycles.
    """
    name = 'ubound'

    def __init__(self, graph, agents, curr_agent_id, budget, diff_model, r):
        super().__init__(graph, agents, curr_agent_id, budget, diff_model, r)

    def run(self):
        # TODO
        pass