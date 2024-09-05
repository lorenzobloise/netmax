import networkx as nx
from src.common import utils
from agent import Agent
import endorsement_policies
#from algorithms import simulation_based
#from algorithms import proxy_based
import diffusion_models
from src.common import influence_probabilities
from tqdm import tqdm
import time

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

class CompetitiveInfluenceMaximization:

    def __init__(self, input_graph: nx.DiGraph, agents: list[Agent],
                 alg: str, diff_model, inf_prob: str = 'uniform',
                 endorsement_policy: endorsement_policies.EndorsementPolicy = endorsement_policies.Random,
                 insert_prob: bool = False, inv_edges: bool = False, r: int = 100):
        """
        Create an instance of the CompetitiveInfluenceMaximization class.
        :param input_graph: A directed graph representing the social network.
        :param agents: A list of Agent instances.
        :param alg: The algorithm to use for influence maximization.
        :param diff_model: The diffusion model to use.
        :param inf_prob: Probability distribution to generate (if needed) the probabilities of influence between nodes. The framework implements different probability distributions, default is 'uniform'.
        :param endorsement_policy: The policy that nodes use to choose which agent to endorse when they have been contacted by more than one agent.
        :param insert_prob: A boolean indicating whether to insert probabilities.
        :param inv_edges: A boolean indicating whether to invert the edges of the graph.
        :param r: Number of simulations to execute. Default is 100.
        """
        self.graph = input_graph.copy()
        self.agents = agents
        # Check if the graph is compatible
        budget = sum([agent.budget for agent in agents])
        n_nodes = len(self.graph.nodes)
        if sum([agent.budget for agent in agents]) > len(self.graph.nodes):
            raise ValueError(
                f"The budget ({budget}) exceeds the number of nodes in the graph ({n_nodes}) by {budget - n_nodes}")

        # Check and set the diffusion model, the algorithm and the influence probabilities
        diff_model_class, alg_class, inf_prob_class, endorsement_policy_class = self.__check_params__(diff_model, alg, inf_prob, endorsement_policy)
        self.inf_prob = inf_prob_class()
        self.endorsement_policy = endorsement_policy_class()
        self.diff_model = diff_model_class(self.endorsement_policy)
        self.mapping = self.__preprocess__()
        self.result = None
        # Set the parameters
        self.insert_prob = insert_prob
        self.inv_edges = inv_edges
        self.r = r
        # Instantiate the algorithm
        self.alg = alg_class(self.graph, self.agents, diff_model_class, inf_prob_class, insert_prob, inv_edges)

    def __check_params__(self, diff_model_name, alg_name, inf_prob_name, endorsement_policy_name):
        """
        Check if the diffusion model,the algorithm and the Probability distribution exist and return the corresponding class.
        :return: The classes of the diffusion model, the algorithm and the influence probabilities.
        :rtype: tuple (diffusion_models.DiffusionModel, algorithm.Algorithm, influence_probabilities.InfluenceProbability, endorsement_policies.EndorsementPolicy)
        """
        hierarchy = (#utils.find_hierarchy(simulation_based.SimulationBasedAlgorithm) |
                     #utils.find_hierarchy(proxy_based.ProxyBasedAlgorithm) |
                     utils.find_hierarchy(diffusion_models.DiffusionModel) |
                     utils.find_hierarchy(influence_probabilities.InfluenceProbability) |
                     utils.find_hierarchy(endorsement_policies.EndorsementPolicy))
        for (k, v) in {'alg': alg_name, 'diff_model': diff_model_name, 'inf_prob': inf_prob_name, 'endorsement_policy': endorsement_policy_name}.items():
            if v not in list(hierarchy.keys()):
                raise ValueError(f"Argument '{v}' not supported for field '{k}'")
        diff_model = hierarchy[diff_model_name]
        alg = hierarchy[alg_name]
        inf_prob = hierarchy[inf_prob_name]
        endorsement_policy = hierarchy[endorsement_policy_name]
        return diff_model, alg, inf_prob, endorsement_policy

    def __preprocess__(self):
        """
        Preprocess the graph before running the multi_agent game.
        First remove isolated nodes and then insert probabilities if needed.
        :return: The mapping between the original nodes and the new nodes.
        """
        mapping = remove_isolated_nodes(self.graph)
        if self.inv_edges:
            self.graph = self.graph.reverse(copy=False)
        if self.insert_prob:
            for (source, target) in self.graph.edges:
                self.graph[source][target]['p'] = self.inf_prob.get_prob(self.graph, source, target)
        return mapping

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
        spreads = {a.name: simulation(graph=self.graph, diff_model=self.diff_model, agents=self.agents, r=self.r) for a in self.agents}
        for a in self.agents:
            a.seed = [inverse_mapping[s] for s in a.seed]
            a.spread = spreads[a.name]
        self.result = {
            'seed': {a: a.seed for a in self.agents},
            'spread': {a: a.spread for a in self.agents},
            'execution_time': execution_time
        }
        return self.result['seed']