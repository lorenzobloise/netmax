from src.single_agent.algorithms.algorithm import Algorithm

class SimulationBasedAlgorithm(Algorithm):
    """
    Simulation-based algorithms for seed set selection in influence maximization problems rely on simulating the spread
    of influence through a network to identify the most influential nodes.
    These algorithms use Monte Carlo simulations to estimate the expected spread of influence for different sets of seed
    nodes. The goal is to select a set of nodes that maximizes the spread of influence within a given budget.
    """

    def __init__(self, graph, agent, budget, diff_model, r):
        super().__init__(graph, agent, budget, diff_model, r)
        self.graph = graph
        self.agent = agent
        self.budget = budget
        self.diff_model = diff_model
        self.r = r

    def run(self):
        raise NotImplementedError("This method must be implemented by subclasses")