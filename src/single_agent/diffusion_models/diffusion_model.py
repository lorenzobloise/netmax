class DiffusionModel:

    def preprocess_data(self, graph):
        raise NotImplementedError("This method must be implemented by subclasses")

    def activate(self, graph, agent, seed):
        raise NotImplementedError("This method must be implemented by subclasses")