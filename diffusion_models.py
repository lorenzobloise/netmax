import random
import im



class DiffusionModel:

    def __init__(self, endorsement_strategy):
        self.endorsement_strategy = endorsement_strategy

    def preprocess_data(self, graph):
        raise NotImplementedError("This method must be implemented by subclasses")

    def activate(self, graph, agent, seed):
        raise NotImplementedError("This method must be implemented by subclasses")

    def competitive_activate(self, graph, agents):
        raise NotImplementedError("This method must be implemented by subclasses")

    def __group_by_agent__(self, graph, active_set):
        dict_result = {}
        for u in active_set:
            curr_agent = graph.nodes[u]['agent']
            if curr_agent in dict_result:
                dict_result[curr_agent].append(u)
            else:
                dict_result[curr_agent] = [u]
        return dict_result


class IndependentCascade(DiffusionModel):
    """
    Paper: Goldenberg et al. - "Talk of the network: A complex system look at the underlying process of word-of-mouth"
    """

    name = 'ic'

    def __init__(self, endorsement_strategy):
        super().__init__(endorsement_strategy)

    def preprocess_data(self, graph):
        return

    def activate(self, graph, agent, seed):
        """
        #TODO maybe broken, should be removed because the new return value will be a dictionary as in competitive_activate
        """
        sim_graph = graph.copy()
        for u in seed:
            im.activate_node(sim_graph, u, agent)
        active_set = im.active_nodes(sim_graph)
        newly_activated = list(active_set)
        old_active_set = []
        while active_set != old_active_set:
            old_active_set = active_set.copy()
            tmp = []
            for u in newly_activated:
                inactive_out_edges = [(u, v, attr) for (u, v, attr) in sim_graph.out_edges(u, data=True) if
                                      not im.is_active(v, sim_graph)]
                for (_, v, attr) in inactive_out_edges:
                    r = random.random()
                    if r < attr['p']:
                        im.activate_node(sim_graph, v, agent)
                        active_set.append(v)
                        tmp.append(v)
            newly_activated = tmp
        return active_set

    def competitive_activate(self, graph, agents):
        """
        :return: A dictionary with the agents as keys and the list of nodes activated by each agent as values
        """
        sim_graph = graph.copy()
        for agent in agents:
            for u in agent.seed:
                im.activate_node(sim_graph, u, agent.name)
        active_set = im.active_nodes(sim_graph)
        newly_activated = list(active_set)
        old_active_set = []
        while active_set != old_active_set:
            old_active_set = active_set.copy()
            # First phase: try to influence inactive nodes
            # Each newly activated node tries to activate its inactive neighbors by contacting them
            for u in newly_activated:
                inactive_out_edges = [(u, v, attr) for (u, v, attr) in sim_graph.out_edges(u, data=True) if
                                      not im.is_active(v, sim_graph)]
                for (_, v, attr) in inactive_out_edges:
                    r = random.random()
                    if r < attr['p']:
                        im.contact_node(sim_graph, v, sim_graph.nodes[u]['agent'])
            # Second phase: contacted inactive nodes choose which agent to endorse by a strategy
            newly_activated = im.manage_pending_nodes(sim_graph, self.endorsement_strategy)
            active_set.extend(newly_activated)

        # Group the activated nodes by agent and return the result
        return self.__group_by_agent__(sim_graph, active_set)


class LinearThreshold(DiffusionModel):
    """
    Paper: Granovetter et al. - "Threshold models of collective behavior"
    """

    name = 'lt'

    def __init__(self, endorsement_strategy):
        super().__init__(endorsement_strategy)

    def preprocess_data(self, graph):
        for node in graph.nodes:
            graph.nodes[node]['threshold'] = random.random()
            graph.nodes[node]['prob_sum'] = 0

    def activate(self, graph, agent, seed):
        sim_graph = graph.copy()
        for u in seed:
            im.activate_node(sim_graph, u, agent)
            for (_, v, attr) in sim_graph.out_edges(u, data=True):
                if not im.is_active(v, sim_graph):
                    sim_graph.nodes[v]['prob_sum'] += attr['p']
        active_set = im.active_nodes(sim_graph)
        old_active_set = []
        while active_set != old_active_set:
            inactive_set = im.inactive_nodes(sim_graph)
            old_active_set = active_set.copy()
            for u in inactive_set:
                if sim_graph.nodes[u]['prob_sum'] >= sim_graph.nodes[u]['threshold']:
                    im.activate_node(sim_graph, u, agent)
                    active_set.append(u)
                    for (_, v, attr) in sim_graph.out_edges(u, data=True):
                        if not im.is_active(v, sim_graph):
                            sim_graph.nodes[v]['prob_sum'] += attr['p']
        return active_set


class Triggering(DiffusionModel):
    """
    Paper: Kempe et al. - "Maximizing the Spread of Influence through a Social Network"
    """

    name = 'tr'

    def __init__(self, endorsement_strategy):
        super().__init__(endorsement_strategy)

    def preprocess_data(self, graph):
        for node in graph.nodes:
            graph.nodes[node]['trigger_set'] = [] # Nodes that 'node' is activated by
            graph.nodes[node]['reverse_trigger_set'] = [] # Nodes that 'node' activates
        for v in graph.nodes:
            in_edges = graph.in_edges(v, data=True)
            for (u, _, edge_attr) in in_edges:
                r = random.random()
                if r < edge_attr['p']:
                    graph.nodes[v]['trigger_set'].append(u)
                    graph.nodes[u]['reverse_trigger_set'].append(v)

    def activate(self, graph, agent, seed):
        sim_graph = graph.copy()
        for u in seed:
            im.activate_node(sim_graph, u, agent)
        active_set = im.active_nodes(sim_graph)
        newly_activated = list(active_set)
        old_active_set = []
        while active_set != old_active_set:
            old_active_set = active_set.copy()
            tmp = []
            for u in newly_activated:
                for v in sim_graph.nodes[u]['reverse_trigger_set']:
                    if not im.is_active(v, sim_graph):
                        im.activate_node(sim_graph, v, agent)
                        active_set.append(v)
                        tmp.append(v)
            newly_activated = tmp
        return active_set


class DecreasingCascade(DiffusionModel):
    """
    Paper: Kempe et al. - "Influential Nodes in a Diffusion Model for Social Networks"
    """

    name = 'dc'

    def __init__(self, endorsement_strategy):
        super().__init__(endorsement_strategy)

    def preprocess_data(self, graph):
        for node in graph.nodes:
            graph.nodes[node]['trials'] = 0

    def activate(self, graph, agent, seed):
        sim_graph = graph.copy()
        for u in seed:
            im.activate_node(sim_graph, u, agent)
        active_set = im.active_nodes(sim_graph)
        newly_activated = list(active_set)
        old_active_set = []
        while active_set != old_active_set:
            old_active_set = active_set.copy()
            tmp = []
            for u in newly_activated:
                inactive_out_edges = [(u, v, attr) for (u, v, attr) in sim_graph.out_edges(u, data=True) if
                                      not im.is_active(v, sim_graph)]
                for (_, v, attr) in inactive_out_edges:
                    r = random.random()
                    trials = sim_graph.nodes[u]['trials']
                    if r < attr['p']*(1/(0.1*(trials**2)+1)):
                        im.activate_node(sim_graph, v, agent)
                        active_set.append(v)
                        tmp.append(v)
                    else:
                        sim_graph.nodes[u]['trials'] = trials + 1
            newly_activated = tmp
        return active_set