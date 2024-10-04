class InfluenceProbability:

    def get_probability(self, graph, u, v):
        raise NotImplementedError("This method must be implemented by subclasses")

    def update_probability(self, graph, u, agent):
        """
        This method is responsible for updating the dynamic features and the related influence probability after a node is activated.
        The only subclasses that overrides this method are the ones that base the probability distribution on dynamic features, such as the opinion of the nodes.
        This method must not be implemented by classes not based on dynamic features because `update_probability` is always called when a node is activated but does nothing in non-dynamic probabilities.
        """
        pass

    def restore_probability(self, graph, u):
        """
        This method is responsible for restoring the dynamic features and the related influence probability after a node is deactivated.
        The only subclasses that overrides this method are the ones that base the probability distribution on dynamic features, such as the opinion of the nodes.
        This method must not be implemented by classes not based on dynamic features because `restore_probability` is always called when a node is deactivated but does nothing in non-dynamic probabilities.
        """
        pass