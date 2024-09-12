class DiffusionModel:

    def __init__(self, endorsement_policy):
        self.endorsement_policy = endorsement_policy
        self.sim_graph = None

    def __copy__(self):
        result = DiffusionModel(self.endorsement_policy)
        if self.sim_graph is not None:
            result.sim_graph = self.sim_graph.copy()
            for key, value in self.sim_graph.graph.items():  # Copy the graph's attributes
                result.sim_graph.graph[key] = value
        return result

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