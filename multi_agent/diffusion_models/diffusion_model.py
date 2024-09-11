class DiffusionModel:

    def __init__(self, endorsement_strategy):
        self.endorsement_strategy = endorsement_strategy

    def preprocess_data(self, graph):
        raise NotImplementedError("This method must be implemented by subclasses")

    def activate(self, graph, agents):
        raise NotImplementedError("This method must be implemented by subclasses")

    def __group_by_agent__(self, graph, active_set):
        dict_result = {}
        for u in active_set:
            curr_agent = graph.nodes[u]['agent'].name
            if curr_agent in dict_result:
                dict_result[curr_agent].append(u)
            else:
                dict_result[curr_agent] = [u]
        return dict_result