import networkx as nx
import utils
from algorithms import simulation_based, proxy_based
import diffusion_models
import influence_probabilities
import time
from tqdm import tqdm
import endorsement_strategies


def activate_node(graph, node, agent_name):
    """
    Activate a node in the graph by setting its status to 'ACTIVE' and the agent name that activated it.
    :param graph: The input graph (networkx.DiGraph).
    :param node: The node to activate.
    :param agent_name: The agent that activates the node.
    """
    graph.nodes[node]['agent'] = agent_name
    graph.nodes[node]['status'] = 'ACTIVE'
    if 'contacted_by' in graph.nodes[node]:
        del graph.nodes[node]['contacted_by']


def contact_node(graph, node, agent_name):
    """
    Contact a node in the graph by setting its status to 'PENDING' and adding the agent name that contacted it.
    :param graph: The input graph (networkx.DiGraph).
    :param node: The node to contact.
    :param agent_name: The agent that contacts the node.
    """
    graph.nodes[node]['status'] = 'PENDING'
    if 'contacted_by' not in graph.nodes[node]:
        graph.nodes[node]['contacted_by'] = set()
    graph.nodes[node]['contacted_by'].add(agent_name)


def manage_pending_nodes(graph, endorsement_strategy):
    newly_activated = []
    for node in pending_nodes(graph):
        chosen_agent_name = endorsement_strategy.choose_agent(node, graph)
        activate_node(graph, node, chosen_agent_name)
        newly_activated.append(node)
    return newly_activated


def active_nodes(graph):
    return [u for u in graph.nodes if is_active(u, graph)]


def inactive_nodes(graph):
    return [u for u in graph.nodes if not is_active(u, graph)]


def pending_nodes(graph):
    return [u for u in graph.nodes if is_pending(u, graph)]


def is_active(node, graph):
    return graph.nodes[node]['status'] == 'ACTIVE'


def is_pending(node, graph):
    return graph.nodes[node]['status'] == 'PENDING'


def invert_edges(graph):
    result = nx.DiGraph()
    nodes = []
    edges = []
    for (node, attr) in graph.nodes(data=True):
        nodes.append((node, attr))
    for (source, target, attr) in graph.edges(data=True):
        edges.append((target, source, attr))
    result.add_nodes_from(nodes)
    result.add_edges_from(edges)
    return result


def remove_isolated_nodes(graph):
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)
    mapping = {old_label: new_label for new_label, old_label in enumerate(graph.nodes)}
    nx.relabel_nodes(graph, mapping, copy=False)
    return mapping


def remove_nodes_not_in_community(community, active_sets):
    """
    Remove nodes that are not in the community from the active sets.
    :param community: The community of nodes.
    :param active_sets: The active sets of the agents. {"agent_name": [active_nodes], ...}
    """
    for agent_name in active_sets.keys():
        active_sets[agent_name] = [node for node in active_sets[agent_name] if node in community]
    return active_sets


def simulation(graph, diff_model, agents, r=10000, community=None):
    spreads = dict()
    for _ in tqdm(range(r), desc="Simulations", position=0, leave=True):
        active_sets = diff_model.activate(graph, agents)
        if community is not None:
            active_sets = remove_nodes_not_in_community(community, active_sets)
        for agent_name in active_sets.keys():
            spreads[agent_name] = spreads.get(agent_name, 0) + len(active_sets[agent_name])
    for agent_name in spreads.keys():
        spreads[agent_name] /= r
    return spreads


# TODO: change agent, seed with agents
def simulation_delta(graph, diff_model, agent, seed1, seed2, r=10000, community=None):
    spread = 0
    for _ in range(r):
        active_set_1 = diff_model.activate(graph, agent, seed1)
        active_set_2 = diff_model.activate(graph, agent, seed2)
        if community is not None:
            active_set_1 = [node for node in active_set_1 if node in community]
            active_set_2 = [node for node in active_set_2 if node in community]
        spread += len([x for x in active_set_1 if x not in active_set_2])
    spread = spread / r
    return spread


class IM:
    class Agent(object):

        def __init__(self, name, budget):
            self.name = name
            self.budget = budget
            self.seed = []
            self.spread = 0

    def __init__(self, input_graph: nx.DiGraph, agents: dict, alg: str = 'celf', diff_model: str = 'ic',
                 inf_prob: str = 'uniform', endorsement_strategy: str = 'random',
                 insert_prob: bool = False, inv_edges: bool = False, r: int = 100):
        """
        Create an instance of the Influence Maximization problem.
        :param input_graph: The input directed graph (networkx.DiGraph).
        :param agents: Dictionary in the form {agent: budget} where 'agent' is a string and 'budget' is an int that represents the number of nodes that particular agent wants to have in its seed set.
        :key alg: Algorithm that solves the maximization problem. The framework implements different algorithms, default is 'celf'.
        :key diff_model: Diffusion model that models the influence spreading among the nodes. The framework implements different diffusion models, default is 'ic'.
        :key inf_prob: Probability distribution to generate (if needed) the probabilities of influence between nodes. The framework implements different probability distributions, default is 'uniform'.
        #:key endorsement_strategy: Strategy that nodes use to endorse a certain agent (only used in a competitive context). The framework implements different endorsement strategies, default is 'random'.
        :key insert_prob: If True, labels the edges with the influence probability.
        :key inv_edges: If True, inverts the graph edges (edge (u,v) becomes (v,u)).
        :key r: Number of simulations to execute. Default is 10000.
        """
        self.input_graph = input_graph
        self.agents = [self.Agent(name, budget) for (name, budget) in agents.items()]
        budget_sum = sum([a.budget for a in self.agents])
        n_nodes = len(self.input_graph.nodes)
        if budget_sum > n_nodes:
            raise ValueError(f"The sum of the budgets ({budget_sum}) exceeds the number of nodes in the graph ({n_nodes}) by {budget_sum - n_nodes}")
        hierarchy = (utils.find_hierarchy(simulation_based.SimulationBasedAlgorithm) |
                     utils.find_hierarchy(proxy_based.ProxyBasedAlgorithm) |
                     utils.find_hierarchy(diffusion_models.DiffusionModel) |
                     utils.find_hierarchy(influence_probabilities.InfluenceProbability) |
                     utils.find_hierarchy(endorsement_strategies.EndorsementStrategy)
                     )
        for (k, v) in {'alg': alg, 'diff_model': diff_model, 'inf_prob': inf_prob}.items():
            if v not in list(hierarchy.keys()):
                raise ValueError(f"Argument '{v}' not supported for field '{k}'")

        self.insert_prob = insert_prob
        self.inv_edges = inv_edges
        self.r = r
        self.graph = self.input_graph.copy()
        self.mapping = None
        self.__preprocess__()
        self.inf_prob = hierarchy[inf_prob]()
        self.diff_model = hierarchy[diff_model](hierarchy[endorsement_strategy])
        self.diff_model.preprocess_data(self.graph)
        self.alg = hierarchy[alg]  # It will be instantiated at execution time
        self.result = None

    def __preprocess__(self):
        self.mapping = remove_isolated_nodes(self.graph)
        if self.inv_edges:
            self.graph = invert_edges(self.graph)
        if self.insert_prob:
            for (source, target) in self.graph.edges:
                self.graph.edges[source, target]['p'] = self.inf_prob.get_probability(self.graph, source, target)
        for node in self.graph.nodes:
            self.graph.nodes[node]['agent'] = 'NONE'
            self.graph.nodes[node]['status'] = 'INACTIVE'

    # TODO: adapt for multiple agents
    def run(self):
        start_time = time.time()
        alg = self.alg(graph=self.graph, agent=None, budget=1, diff_model=self.diff_model, r=self.r)
        while not self.__game_over__():
            for curr_agent in self.__get_agents_not_fulfilled__():
                alg.set_agent(curr_agent.name)
                alg.set_graph(self.graph)
                partial_seed = alg.run()
                if len(partial_seed) == 0:
                    raise RuntimeError(
                        f"No more available nodes to add to the seed set of agent {curr_agent.name}. Budget not fulfilled by {curr_agent.budget - len(curr_agent.seed)}")
                for node in partial_seed:
                    activate_node(graph=self.graph, node=node, agent_name=curr_agent.name)
                curr_agent.seed.extend(partial_seed)
        execution_time = time.time() - start_time
        inverse_mapping = {new_label: old_label for (old_label, new_label) in self.mapping.items()}
        for a in self.agents:
            a.seed = [inverse_mapping[s] for s in a.seed]
            #a.spread = simulation(graph=self.graph, diff_model=self.diff_model, agent_name=a.name, seed=a.seed, r=self.r) # TODO: change
        self.result = {
            'seed': {a: a.seed for a in self.agents},
            'spread': {a: a.spread for a in self.agents},
            'execution_time': execution_time
        }
        return self.result['seed']

    def __budget_fulfilled__(self, agent):
        """
        Check if the budget of an agent is fulfilled.

        """
        return len(agent.seed) >= agent.budget

    def __get_agents_not_fulfilled__(self):
        """
        Get the agents that have not fulfilled their budget yet.
        :return: List of objects of type Agent that have not fulfilled their budget yet
        """
        return [a for a in self.agents if not self.__budget_fulfilled__(a)]

    def __game_over__(self):
        """
        Check if the game is over.
        :return: True if the game is over, False otherwise
        """
        return all([self.__budget_fulfilled__(a) for a in self.agents])
