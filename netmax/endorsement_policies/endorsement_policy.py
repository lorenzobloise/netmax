class EndorsementPolicy:

    def __init__(self, graph):
        return

    def choose_agent(self, node, sim_graph):
        """
        This method implements the logic behind any endorsement policy and has to be implemented by the subclasses.
        :param node: the node which has to choose an agent to endorse.
        :param sim_graph: the simulation graph.
        :return: the agent to endorse.
        """
        raise NotImplementedError("This method must be implemented by subclasses")