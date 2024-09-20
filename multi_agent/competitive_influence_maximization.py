import networkx as nx
from common import utils
from multi_agent.agent import Agent
from multi_agent.algorithms.algorithm import Algorithm
from multi_agent.endorsement_policies.endorsement_policy import EndorsementPolicy
from  multi_agent.diffusion_models.diffusion_model import DiffusionModel
from common.influence_probabilities.influence_probability import InfluenceProbability
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def activate_node(graph, node, agent: Agent):
    """
    Activate a node in the graph by setting its status to 'ACTIVE' and the agent name that activated it.
    :param graph: The input graph (networkx.DiGraph).
    :param node: The node to activate.
    :param agent: The agent that activates the node.
    """
    graph.nodes[node]['agent'] = agent
    graph.nodes[node]['status'] = 'ACTIVE'
    if 'contacted_by' in graph.nodes[node]:
        del graph.nodes[node]['contacted_by']
    if graph.graph['inf_prob'] is not None:
        graph.graph['inf_prob'].update_probability(graph, node)

def deactivate_node(graph, node):
    """
    Deactivate a node in the graph by setting its status to 'INACTIVE' and deleting the agent name.
    :param graph: The input graph (networkx.DiGraph).
    :param node: The node to deactivate.
    """
    graph.nodes[node]['status'] = 'INACTIVE'
    if 'agent' in graph.nodes[node].keys():
        del graph.nodes[node]['agent']
    if graph.graph['inf_prob'] is not None:
        graph.graph['inf_prob'].update_probability(graph, node)

def contact_node(graph, node, agent: Agent):
    """
    Contact a node in the graph by setting its status to 'PENDING' and adding the agent name that contacted it.
    :param graph: The input graph (networkx.DiGraph).
    :param node: The node to contact.
    :param agent: The agent that contacts the node.
    """
    graph.nodes[node]['status'] = 'PENDING'
    if 'contacted_by' not in graph.nodes[node]:
        graph.nodes[node]['contacted_by'] = set()
    graph.nodes[node]['contacted_by'].add(agent)

def manage_pending_nodes(graph, endorsement_policy, pending_nodes_list):
    newly_activated = []
    for node in pending_nodes_list:
        contacted_by = graph.nodes[node]['contacted_by']
        chosen_agent = endorsement_policy.choose_agent(node, graph) if len(contacted_by) > 1 else next(iter(contacted_by))
        activate_node(graph, node, chosen_agent)
        newly_activated.append(node)
    return newly_activated

def active_nodes(graph: nx.DiGraph):
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

def concurrent_simulation(graph, agents, diff_model, r):
    spreads = dict()
    for i in range(r):
        active_sets = diff_model.activate(graph, agents)
        for agent_name in active_sets.keys():
            spreads[agent_name] = spreads.get(agent_name, 0) + len(active_sets[agent_name])
    result = spreads
    return result

def simulation_old(graph, diff_model, agents, r=10000, community=None):
    spreads = dict()
    num_threads = 8
    with ProcessPoolExecutor() as exe:
        futures = []
        for i in range(num_threads):
            srange = int((i + 1) * r / num_threads)-int(i * r / num_threads)
            future = exe.submit(concurrent_simulation, graph, agents, diff_model.__copy__(), srange)
            futures.append(future)
        for f in as_completed(futures):
            result = f.result()
            for agent_name in result.keys():
                spreads[agent_name] = spreads.get(agent_name, 0) + result[agent_name]
    for agent_name in spreads.keys():
        spreads[agent_name] /= r
    return spreads

def __log_simulation__(logger, i, r):
    if i % (r//4)==0:
        logger.info(f"Simulations: {(i / r) * 100}%")

def simulation(graph, diff_model, agents, r=10000, community=None, verbose=False):
    spreads = dict()
    progress_bar = None
    if verbose:
        progress_bar = tqdm(total=r)
    for _ in (range(r)):
        active_sets = diff_model.activate(graph, agents)
        if community is not None:
            active_sets = remove_nodes_not_in_community(community, active_sets)
        for agent_name in active_sets.keys():
            spreads[agent_name] = spreads.get(agent_name, 0) + len(active_sets[agent_name])
        if verbose:
            progress_bar.update()
    for agent_name in spreads.keys():
        spreads[agent_name] /= r
    return spreads

def simulation_delta(graph, diff_model, agents, curr_agent_id, seed1, seed2, r=10000, community=None):
    """
    Computes the spread as follows:
    1) Computes the activated nodes from the first seed set {active_set_1}
    2) Computes the activated nodes from the second seed set {active_set_2}
    3) Returns a dictionary containing the average spread (on r experiments) of {active_set_1}-{active_set_2} for each agent
    """
    spreads = dict()
    for _ in range(r):
        old_seed_set = agents[curr_agent_id].__getattribute__('seed')
        agents[curr_agent_id].__setattr__('seed', seed1)
        active_sets_1 = diff_model.activate(graph, agents)
        # Set agents curr agent a seed 2
        agents[curr_agent_id].__setattr__('seed', seed2)
        active_sets_2 = diff_model.activate(graph, agents)
        # Restore old seed set
        agents[curr_agent_id].__setattr__('seed', old_seed_set)
        if community is not None:
            for agent_name in active_sets_1.keys():
                active_sets_1[agent_name] = [node for node in active_sets_1[agent_name] if node in community]
            for agent_name in active_sets_2.keys():
                active_sets_2[agent_name] = [node for node in active_sets_2[agent_name] if node in community]
        active_sets = dict()
        for agent in agents:
            active_sets[agent.name] = [x for x in active_sets_1[agent.name] if x not in active_sets_2[agent.name]]
            spreads[agent.name] = spreads.get(agent.name, 0) + len(active_sets[agent.name])
    for agent_name in spreads.keys():
        spreads[agent_name] = spreads[agent_name] / r
    return spreads

class CompetitiveInfluenceMaximization:

    def __init__(self, input_graph: nx.DiGraph, agents: list[Agent],
                 alg: str, diff_model, inf_prob: str = None, endorsement_policy: str = 'random',
                 insert_opinion: bool = False, inv_edges: bool = False, r: int = 100):
        """
        Create an instance of the CompetitiveInfluenceMaximization class.
        :param input_graph: A directed graph representing the social network.
        :param agents: A list of Agent instances.
        :param alg: The algorithm to use for influence maximization.
        :param diff_model: The diffusion model to use.
        :param inf_prob: Probability distribution to generate (if needed) the probabilities of influence between nodes. The framework implements different probability distributions, default is None.
        :param endorsement_policy: The policy that nodes use to choose which agent to endorse when they have been contacted by more than one agent. The framework implements different endorsement policies, default is 'random'.
        :param insert_opinion: True if the nodes do not contain an information about their opinion on the agents, False otherwise or if the opinion is not used.
        :param inv_edges: A boolean indicating whether to invert the edges of the graph.
        :param r: Number of simulations to execute. Default is 100.
        """
        self.graph = input_graph.copy() # TODO
        self.agents = agents
        # Check if the graph is compatible
        budget = sum([agent.budget for agent in agents])
        n_nodes = len(self.graph.nodes)
        if sum([agent.budget for agent in agents]) > len(self.graph.nodes):
            raise ValueError(f"The budget ({budget}) exceeds the number of nodes in the graph ({n_nodes}) by {budget - n_nodes}")
        # Check and set the diffusion model, the algorithm and the influence probabilities
        diff_model_class, alg_class, inf_prob_class, endorsement_policy_class = self.__check_params__(diff_model, alg, inf_prob, endorsement_policy)
        self.inf_prob = None if inf_prob_class is None else inf_prob_class()
        self.endorsement_policy = endorsement_policy_class(self.graph)
        self.diff_model = diff_model_class(self.endorsement_policy)
        self.graph.graph['inf_prob'] = self.inf_prob
        self.graph.graph['insert_opinion'] = insert_opinion
        # Set the parameters
        self.insert_opinion = insert_opinion
        self.inv_edges = inv_edges
        self.mapping = self.__preprocess__()
        self.result = None
        self.r = r
        self.diff_model.preprocess_data(self.graph)
        # Instantiate the algorithm
        #self.alg = alg_class(self.graph, self.agents, diff_model_class, inf_prob_class, insert_prob, inv_edges)
        self.alg = alg_class
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def __check_params__(self, diff_model_name, alg_name, inf_prob_name, endorsement_policy_name):
        """
        Check if the diffusion model,the algorithm and the Probability distribution exist and return the corresponding class.
        :return: The classes of the diffusion model, the algorithm and the influence probabilities.
        :rtype: tuple (diffusion_models.DiffusionModel, algorithm.Algorithm, influence_probabilities.InfluenceProbability, endorsement_policies.EndorsementPolicy)
        """
        hierarchy: dict = dict(utils.find_hierarchy(Algorithm) +
                               utils.find_hierarchy(DiffusionModel) +
                               utils.find_hierarchy(InfluenceProbability) +
                               utils.find_hierarchy(EndorsementPolicy))
        hierarchy[None] = None
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
        for node in self.graph.nodes:
            self.graph.nodes[node]['status'] = 'INACTIVE'
            if self.insert_opinion:
                self.graph.nodes[node]['opinion'] = [1/len(self.agents) for _ in self.agents]
        if self.inf_prob is not None:
            for (source, target) in self.graph.edges:
                self.graph[source][target]['p'] = self.inf_prob.get_probability(self.graph, source, target)
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
        alg = self.alg(graph=self.graph, agents=self.agents, curr_agent_id=None, budget=1, diff_model=self.diff_model, r=self.r)
        self.logger.info("Starting the competitive influence maximization game")
        round_counter = 0
        while not self.__game_over__():
            self.logger.info(f"Round {round_counter} has started")
            for agent in self.__get_agents_not_fulfilled__():
                self.logger.info(f"Agent {agent.name} (id: {agent.id}) is playing")
                alg.set_curr_agent(agent.id)
                alg.set_graph(self.graph)
                partial_seed, new_spreads = alg.run()
                self.logger.info(f"The Spreads of agents {new_spreads}")
                if len(partial_seed) == 0:
                    raise RuntimeError(
                        f"No more available nodes to add to the seed set of agent {agent.name}. Budget not fulfilled by {agent.budget - len(agent.seed)}")
                for node in partial_seed:
                    self.logger.info(f"Activating node {node} by agent {agent.name}")
                    activate_node(graph=self.graph, node=node, agent=agent)
                for a in self.agents:
                    a.spread = new_spreads[a.name]
                self.logger.info(f"Spread of agent {agent.name} updated with {agent.spread} after adding node {partial_seed}")
                agent.seed.extend(partial_seed)
                self.logger.info(f"Seed set of agent {agent.name} updated with {partial_seed[0]} node")
            round_counter += 1
        self.logger.info(f"Game over")
        inverse_mapping = {new_label: old_label for (old_label, new_label) in self.mapping.items()}
        self.logger.info(f"Seed sets found:")
        for a in self.agents:
            self.logger.info(f"{a.name}: {[inverse_mapping[s] for s in a.seed]}")
        self.logger.info(f"Starting the spreads estimation with {self.r} simulations")
        execution_time = time.time() - start_time
        spreads = simulation(graph=self.graph, diff_model=self.diff_model, agents=self.agents, r=self.r, verbose=True)
        for a in self.agents:
            a.seed = [inverse_mapping[s] for s in a.seed]
            a.spread = spreads[a.name]
        self.result = {
            'seed': {a.name: a.seed for a in self.agents},
            'spread': {a.name: a.spread for a in self.agents},
            'execution_time': execution_time
        }
        return self.result['seed']