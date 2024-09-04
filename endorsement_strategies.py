import random


class EndorsementStrategy:

    def choose_agent(self, node, graph):
        raise NotImplementedError("This method must be implemented by subclasses")


class RandomEndorsementStrategy(EndorsementStrategy):

    name = "random"

    def choose_agent(self, node, graph):
        return random.choice(graph.nodes[node]['contacted_by'])


# TODO: other strategies