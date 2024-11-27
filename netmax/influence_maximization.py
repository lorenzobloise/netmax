import copy
import random
import networkx as nx
from netmax.agent import Agent
from netmax.algorithms.algorithm import Algorithm
from netmax.endorsement_policies.endorsement_policy import EndorsementPolicy
from netmax.diffusion_models.diffusion_model import DiffusionModel
from netmax.influence_probabilities.influence_probability import InfluenceProbability
import time
import logging
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
    # Remove the 'contacted_by' key-value pair from the node's attributes to avoid memory waste
    if 'contacted_by' in graph.nodes[node]:
        del graph.nodes[node]['contacted_by']
    # Update the influence probabilities (only works in a dynamic probabilities setting, i.e. opinion-based)
    if graph.graph['inf_prob'] is not None:
        graph.graph['inf_prob'].update_probability(graph, node, agent)

def activate_node_in_simulation_graph(graph, sim_graph, node, agent: Agent):
    """
    Activate node in the simulation graph but update (temporarily) the influence probabilities in the original graph.
    :param graph: The original graph (networkx.DiGraph).
    :param sim_graph: The simulation graph (networkx.DiGraph).
    :param node: The node to activate.
    :param agent: The agent that activates the node.
    """
    sim_graph.nodes[node]['agent'] = agent
    sim_graph.nodes[node]['status'] = 'ACTIVE'
    # Remove the 'contacted_by' key-value pair from the node's attributes to avoid memory waste
    if 'contacted_by' in sim_graph.nodes[node]:
        del sim_graph.nodes[node]['contacted_by']
    # Update the influence probabilities (only works in a dynamic probabilities setting, i.e. opinion-based)
    if graph.graph['inf_prob'] is not None:
        graph.graph['inf_prob'].update_probability(graph, node, agent)

def deactivate_node(graph, node):
    """
    Deactivate a node in the graph by setting its status to 'INACTIVE' and deleting the agent name.
    :param graph: The input graph (networkx.DiGraph).
    :param node: The node to deactivate.
    """
    graph.nodes[node]['status'] = 'INACTIVE'
    # Remove the 'agent' key-value pair from the node's attributes to avoid memory waste
    if 'agent' in graph.nodes[node].keys():
        del graph.nodes[node]['agent']
    # Restore the influence probabilities (only works in a dynamic probabilities setting, i.e. opinion-based)
    if graph.graph['inf_prob'] is not None:
        graph.graph['inf_prob'].restore_probability(graph, node)

def deactivate_node_in_simulation_graph(graph, sim_graph, node):
    """
    Deactivate node in the simulation graph but restore (temporarily) the influence probabilities in the original graph.
    :param graph: The input graph (networkx.DiGraph).
    :param sim_graph: The simulation graph (networkx.DiGraph).
    :param node: The node to deactivate.
    """
    sim_graph.nodes[node]['status'] = 'INACTIVE'
    # Remove the 'agent' key-value pair from the node's attributes to avoid memory waste
    if 'agent' in sim_graph.nodes[node].keys():
        del sim_graph.nodes[node]['agent']
    # Restore the influence probabilities (only works in a dynamic probabilities setting, i.e. opinion-based)
    if graph.graph['inf_prob'] is not None:
        graph.graph['inf_prob'].restore_probability(graph, node)

def contact_node(graph, node, agent: Agent):
    """
    Contact a node in the graph by setting its status to 'PENDING' and adding the agent name that contacted it.
    :param graph: The input graph (networkx.DiGraph).
    :param node: The node to contact.
    :param agent: The agent that contacts the node.
    """
    graph.nodes[node]['status'] = 'PENDING'
    # Add this agent to the ones that contacted this node
    if 'contacted_by' not in graph.nodes[node]:
        graph.nodes[node]['contacted_by'] = set()
    graph.nodes[node]['contacted_by'].add(agent)

def manage_pending_nodes(graph, sim_graph, endorsement_policy, pending_nodes_list):
    """
    Second step of the activation process (in progressive diffusion models):
    the nodes who have been contacted by some agent (and thus are in PENDING state)
    need to choose which agent to endorse using some endorsement policy specified by the user. Then, they become
    ACTIVE for the agent they chose.
    :param graph: The input graph (networkx.DiGraph).
    :param sim_graph: The simulation graph (networkx.DiGraph).
    :param endorsement_policy: The endorsement policy specified by the user.
    :param pending_nodes_list: The list of nodes who have been contacted by some agent.
    :return: The list of the newly activated nodes.
    """
    newly_activated = []
    for node in pending_nodes_list:
        contacted_by = sim_graph.nodes[node]['contacted_by']
        # Use the endorsement policy to choose the agent, but if there's only one agent the choice is obvious
        chosen_agent = endorsement_policy.choose_agent(node, sim_graph) if len(contacted_by) > 1 else next(iter(contacted_by))
        # Once decided the agent, activate the node inside the simulation graph
        activate_node_in_simulation_graph(graph, sim_graph, node, chosen_agent)
        newly_activated.append(node)
    return newly_activated

def put_node_into_quiescent(graph, node, agent: Agent):
    """
    Only works for the F2DLT diffusion models. Once the node has chosen which agent
    to endorse, they don't become ACTIVE straight away, but instead become QUIESCENT, and after the quiescence time
    (computed according to the model specifications) they become ACTIVE.
    :param graph: The input graph (networkx.DiGraph).
    :param node: The node to put into QUIESCENT state.
    :param agent: The agent which the node has endorsed.
    """
    graph.nodes[node]['status'] = 'QUIESCENT'
    graph.nodes[node]['agent'] = agent
    # Remove the 'contacted_by' key-value pair from the node's attributes to avoid memory waste
    if 'contacted_by' in graph.nodes[node]:
        del graph.nodes[node]['contacted_by']

def transition_nodes_into_quiescent_state(sim_graph, endorsement_policy, pending_nodes_list):
    """
    Only works for the F2DLT diffusion models. The nodes which are in PENDING state need to choose which agent to endorse,
    and they do it using the endorsement policy (which has been specified by the user), then they enter the QUIESCENT
    state.
    :param sim_graph: The simulation graph (networkx.DiGraph).
    :param endorsement_policy: The endorsement policy specified by the user.
    :param pending_nodes_list: The list of nodes who have been contacted by some agent.
    :return: The list of the nodes which have become QUIESCENT.
    """
    quiescent_nodes = []
    for node in pending_nodes_list:
        contacted_by = sim_graph.nodes[node]['contacted_by']
        # Use the endorsement policy to choose the agent, but if there's only one agent the choice is obvious
        chosen_agent = endorsement_policy.choose_agent(node, sim_graph) if len(contacted_by) > 1 else next(iter(contacted_by))
        # Once decided the agent, put the node into quiescent state inside the simulation graph
        put_node_into_quiescent(sim_graph, node, chosen_agent)
        quiescent_nodes.append(node)
    return quiescent_nodes

def active_nodes(graph: nx.DiGraph):
    """
    Returns the nodes which are ACTIVE in the input graph.
    :param graph: The input graph (networkx.DiGraph).
    :return: The list of nodes which are ACTIVE in the input graph.
    """
    return [u for u in graph.nodes if is_active(u, graph)]

def inactive_nodes(graph):
    """
    Returns the nodes which are INACTIVE in the input graph.
    :param graph: The input graph (networkx.DiGraph).
    :return: The list of nodes which are INACTIVE in the input graph.
    """
    return [u for u in graph.nodes if not is_active(u, graph)]

def pending_nodes(graph):
    """
    Returns the nodes which are PENDING in the input graph.
    :param graph: The input graph (networkx.DiGraph).
    :return: The list of nodes which are PENDING in the input graph.
    """
    return [u for u in graph.nodes if is_pending(u, graph)]

def is_active(node, graph):
    """
    Returns True if the node is ACTIVE in the input graph.
    :param node: The node to check.
    :param graph: The input graph (networkx.DiGraph).
    :return: True if the node is ACTIVE in the input graph.
    """
    return graph.nodes[node]['status'] == 'ACTIVE'

def is_pending(node, graph):
    """
    Returns True if the node is PENDING in the input graph.
    :param node: The node to check.
    :param graph: The input graph (networkx.DiGraph).
    :return: True if the node is PENDING in the input graph.
    """
    return graph.nodes[node]['status'] == 'PENDING'

def is_quiescent(node, graph):
    """
    Returns True if the node is QUIESCENT in the input graph.
    :param node: The node to check.
    :param graph: The input graph (networkx.DiGraph).
    :return: True if the node is QUIESCENT in the input graph.
    """
    return graph.nodes[node]['status'] == 'QUIESCENT'

def graph_is_signed(graph):
    """
    Returns True if the input graph's attribute 'signed' is set at True.
    :param graph: The input graph (networkx.DiGraph).
    :return: True if the input graph's attribute 'signed' is set at True.
    """
    return graph.graph['signed']

def build_trust_and_distrust_graphs(graph, verbose=False):
    """
    From the input graph (which is a signed network), build its corresponding trust and distrust subnetworks. Both contain
    all the nodes in the original graph, but the former is built by using only the positive edges, while the latter
    is built by using only the negative edges.
    :param graph: The input graph (networkx.DiGraph).
    :param verbose: If True, displays a progress bar.
    :return: The trust and distrust subnetworks.
    """
    trust_graph = nx.DiGraph()
    distrust_graph = nx.DiGraph()
    for key, value in graph.graph.items():  # Copy the graph's attributes
        trust_graph.graph[key] = value
        distrust_graph.graph[key] = value
    graph_nodes = graph.nodes(data=True)
    progress_bar = None
    if verbose:
        progress_bar = tqdm(total=len(graph.edges), desc="Building trust and distrust graphs")
    for u, v, attr in graph.edges(data=True):
        node_u = graph_nodes[u]
        node_v = graph_nodes[v]
        # Add the nodes that are incident on all the edges, and I'm sure that
        # the result graphs will have all the nodes in the original graph because
        # the original graph has undergone preprocessing, which has deleted isolated nodes
        trust_graph.add_node(u, **node_u)
        trust_graph.add_node(v, **node_v)
        distrust_graph.add_node(u, **node_u)
        distrust_graph.add_node(v, **node_v)
        if attr['p'] > 0: # v trusts u
            trust_graph.add_edge(u, v, **attr)
        else:
            distrust_graph.add_edge(u, v, **attr)
        if verbose:
            progress_bar.update(1)
    return trust_graph, distrust_graph

def remove_isolated_nodes(graph):
    """
    Removes all isolated nodes from the input graph because they don't contribute actively to the influence propagation
    (no edge is incident on them). After removing the isolated nodes, this method changes the labels of the remaining nodes,
    so that they are ordered from 0. The correspondence (old label -> new label) is stored inside a mapping dictionary,
    which is returned along with the graph.
    :param graph: The input graph (networkx.DiGraph).
    :return: The mapping used for relabeling the remaining nodes and the updated graph.
    """
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)
    mapping = {old_label: new_label for new_label, old_label in enumerate(graph.nodes)}
    graph = nx.relabel_nodes(graph, mapping)
    return mapping, graph

def simulation(graph, diff_model, agents, r, verbose=False):
    """
    Simulates r times the diffusion process inside the input graph according to the given diffusion model.
    :param graph: The input graph (networkx.DiGraph).
    :param diff_model: The diffusion model.
    :param agents: The list of agents.
    :param r: The number of simulations.
    :param verbose: If True, displays a progress bar.
    :return: A dictionary containing the average spread for each agent.
    """
    spreads = dict() # Dictionary <agent_name>: <spread>
    progress_bar = None
    if verbose:
        progress_bar = tqdm(total=r, desc="Simulations")
    for _ in (range(r)):
        active_sets = diff_model.activate(graph, agents) # This is the single simulation
        for agent_name in active_sets.keys():
            spreads[agent_name] = spreads.get(agent_name, 0) + len(active_sets[agent_name]) # Update the sum of the spreads for each agent
        if verbose:
            progress_bar.update()
    for agent_name in spreads.keys(): # Compute the average spread for each agent
        spreads[agent_name] /= r
    return spreads

def simulation_delta(graph, diff_model, agents, curr_agent_id, seed1, seed2, r):
    """
    Computes the spread as follows. For r experiments:
    1) Computes the activated nodes from the first seed set {active_set_1}
    2) Computes the activated nodes from the second seed set {active_set_2}
    3) Stores the spread of this experiment as |{active_set_1} - {active_set_2}|
    Then returns a dictionary containing the average spread for each agent.
    :param graph: The input graph (networkx.DiGraph).
    :param diff_model: The diffusion model.
    :param agents: The list of agents.
    :param curr_agent_id: The current agent id.
    :param seed1: The first seed set.
    :param seed2: The second seed set.
    :param r: The number of simulations.
    :return: A dictionary containing the average spread for each agent.
    """
    spreads = dict() # Dictionary <agent_name>: <spread>
    for _ in range(r):
        old_seed_set = agents[curr_agent_id].__getattribute__('seed')
        # First compute the activated nodes with the first seed set
        agents[curr_agent_id].__setattr__('seed', seed1)
        active_sets_1 = diff_model.activate(graph, agents)
        # Then compute the activated nodes with the second seed set
        agents[curr_agent_id].__setattr__('seed', seed2)
        active_sets_2 = diff_model.activate(graph, agents)
        # Restore old seed set
        agents[curr_agent_id].__setattr__('seed', old_seed_set)
        active_sets = dict()
        # Subtract the two active sets and update the spread of the agents as the size of the result of this operation
        for agent in agents:
            active_sets[agent.name] = [x for x in active_sets_1[agent.name] if x not in active_sets_2[agent.name]]
            spreads[agent.name] = spreads.get(agent.name, 0) + len(active_sets[agent.name])
    # Compute the average spread for each agent
    for agent_name in spreads.keys():
        spreads[agent_name] = spreads[agent_name] / r
    return spreads

class InfluenceMaximization:

    def __init__(self, input_graph: nx.DiGraph, agents: dict,
                 alg: str, diff_model, inf_prob: str = None, endorsement_policy: str = 'random',
                 insert_opinion: bool = False, inv_edges: bool = False, first_random_seed: bool = False, r: int = 100, verbose: bool = False):
        """
        Create an instance of the InfluenceMaximization class.
        :param input_graph: A directed graph representing the network (of type networkx.DiGraph).
        :param agents: A dictionary where the key is the agent name and the value is his budget.
        :param alg: The algorithm to use for influence maximization.
        :param diff_model: The diffusion model to use.
        :param inf_prob: Probability distribution used to generate (if needed) the probabilities of influence between nodes. The framework implements different influence probabilities, default is None.
        :param endorsement_policy: The policy that nodes use to choose which agent to endorse when they have been contacted by more than one agent. The framework implements different endorsement policies, default is 'random'.
        :param insert_opinion: True if the nodes do not contain any information about their opinion on the agents, False otherwise or if the opinion is not used.
        :param inv_edges: A boolean indicating whether to invert the edges of the graph.
        :param first_random_seed: A boolean indicating whether to insert a first node (chosen randomly) in the seed set of every agent.
        :param r: Number of simulations to execute. Default is 100.
        :param verbose: If True sets the logging level to INFO, otherwise displays only the minimal information.
        """
        self.graph = input_graph.copy()
        # Build agents list
        self.agents = []
        for idx, agent_name in enumerate(agents):
            if agents[agent_name] <= 0:
                raise ValueError(f"Agents budgets must be positive")
            self.agents.append(Agent(list(agents.keys())[idx], list(agents.values())[idx], idx))
        self.verbose = verbose
        # Check and set the diffusion model, the algorithm and the influence probabilities
        diff_model_class, alg_class, inf_prob_class, endorsement_policy_class = self.__check_params__(diff_model, alg, inf_prob, endorsement_policy)
        self.inf_prob = None if inf_prob_class is None else inf_prob_class()
        self.endorsement_policy = endorsement_policy_class(self.graph)
        self.diff_model = diff_model_class(self.endorsement_policy)
        # Set the parameters
        self.insert_opinion = insert_opinion
        self.first_random_seed = first_random_seed
        self.inv_edges = inv_edges
        # Pre-process the graph, removing isolated nodes that do not contribute to influence diffusion process
        self.mapping = self.__preprocess__()
        # Check if the graph is compatible (the sum of the budgets must not exceed the number of nodes in the graph)
        # The term 'already_chosen_nodes' takes into account the possibility that the agents have already one node
        # in the seed set, chosen randomly before the beginning of the game (first_random_seed is True in this case)
        budget_sum = sum([agent.budget for agent in self.agents])
        n_nodes = len(self.graph.nodes)
        already_chosen_nodes = self.first_random_seed * len(self.agents)
        if budget_sum > n_nodes - already_chosen_nodes:
            raise ValueError(
                f"The budget ({budget_sum}) exceeds the number of available nodes in the graph ({n_nodes - already_chosen_nodes}) by {budget_sum - n_nodes + already_chosen_nodes}. "
                f"Check the budget for every agent, the number of nodes in the graph and the parameter 'first_random_seed'.")
        self.inverse_mapping = {new_label: old_label for (old_label, new_label) in self.mapping.items()}
        self.result = None
        self.r = r
        self.diff_model.preprocess_data(self.graph)
        self.alg = alg_class
        # Set logging level
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %msg')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.verbose:
            self.logger.propagate = False
        # Dictionary <round_number: agents> containing, for each round, the status of the game at that moment,
        # represented by the 'agents' dictionary
        self.history = {}

    def __check_params__(self, diff_model_name, alg_name, inf_prob_name, endorsement_policy_name):
        """
        Check if the diffusion model, the algorithm and the influence probability exist in the namespace and return the corresponding class.
        :return: The classes of the diffusion model, the algorithm and the influence probability.
        """
        # Build the hierarchy of the namespace
        hierarchy: dict = dict(find_hierarchy(Algorithm) +
                               find_hierarchy(DiffusionModel) +
                               find_hierarchy(InfluenceProbability) +
                               find_hierarchy(EndorsementPolicy))
        hierarchy[None] = None
        # Check if the names exist
        for (k, v) in {'alg': alg_name, 'diff_model': diff_model_name, 'inf_prob': inf_prob_name, 'endorsement_policy': endorsement_policy_name}.items():
            if v not in list(hierarchy.keys()):
                raise ValueError(f"Argument '{v}' not supported for field '{k}'")
        # Return the classes
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
        # Set some attributes of the graph
        self.graph.graph['inf_prob'] = self.inf_prob
        self.graph.graph['insert_opinion'] = self.insert_opinion
        self.graph.graph['signed'] = True
        # If one edge doesn't have the 's' attribute, it means that the graph is not signed.
        # So to verify if a graph is signed we have to check if every single edge has the 's' attribute.
        for _, _, attr in self.graph.edges(data=True):
            if 's' not in attr:
                self.graph.graph['signed'] = False
                break
        # Remove the isolated nodes from the graph, as they do not contribute to the influence spread
        mapping, new_graph = remove_isolated_nodes(self.graph)
        self.graph = new_graph
        # If inv_edges is True, invert all the edges
        if self.inv_edges:
            self.graph = self.graph.reverse(copy=False)
        # Initialize node status
        for node in self.graph.nodes:
            self.graph.nodes[node]['status'] = 'INACTIVE'
            # If insert_opinion is True, initialize the nodes opinion uniformly for every agent
            if self.insert_opinion:
                self.graph.nodes[node]['opinion'] = [1/len(self.agents) for _ in self.agents]
        # If inf_prob is not None, initialize the influence probabilities for each edge with the specified function
        if self.inf_prob is not None:
            for (source, target) in self.graph.edges:
                self.graph[source][target]['p'] = self.inf_prob.get_probability(self.graph, source, target)
        return mapping

    def get_diff_model(self):
        """
        :return: The diffusion model.
        """
        return self.diff_model

    def get_agents(self):
        """
        :return: The 'agents' dictionary.
        """
        return self.agents

    def get_graph(self):
        """
        :return: The graph.
        """
        return self.graph

    def get_history(self):
        """
        :return: The history.
        """
        return self.history

    def get_algorithm_name(self):
        """
        :return: The name of the algorithm class.
        """
        # Check if self.alg is not instantiated yet
        if isinstance(self.alg, type):
            return self.alg.__name__
        return self.alg.__class__.__name__

    def get_diffusion_model_name(self):
        """
        :return: The name of the diffusion model class.
        """
        return self.diff_model.__class__.__name__

    def get_endorsement_policy_name(self):
        """
        :return: The name of the endorsement policy class.
        """
        return self.endorsement_policy.__class__.__name__

    def __budget_fulfilled__(self, agent):
        """
        Check if the budget of an agent is fulfilled.
        """
        return len(agent.seed) >= agent.budget + 1 * self.first_random_seed

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

    def __insert_first_random_seed__(self):
        """
        If the parameter first_random_seed is True, the first seed of each agent is randomly chosen.
        """
        for agent in self.agents:
            node = random.choice(list(self.graph.nodes))
            agent.seed.append(node)
            activate_node(self.graph, node, agent)

    def __register_history__(self, turn_id, current_state):
        """
        This method registers the current state of the game for every turn, to build a history of the whole game.
        """
        self.history[turn_id] = copy.deepcopy(current_state)

    def run(self):
        # Measure the time taken to finish the game
        start_time = time.time()
        # Initialize the algorithm. The budget is set to 1 because at each turn only one node is chosen
        alg = self.alg(graph=self.graph, agents=self.agents, curr_agent_id=None, budget=1, diff_model=self.diff_model, r=self.r)
        self.logger.info(f"Starting influence maximization process with algorithm {alg.__class__.__name__}")
        # If first_random_seed is True, select a random node to add to the agents' seed set
        if self.first_random_seed:
            self.__insert_first_random_seed__()
        self.__register_history__(0, self.agents) # Register the initial state
        round_counter = 1
        turn_counter = 1
        # This piece of code repeats until all agents have fulfilled their budget
        while not self.__game_over__():
            self.logger.info(f"Round {round_counter} has started")
            # Every agent that hasn't already fulfilled its budget executes a single iteration of the algorithm
            # and select the next node to add to its seed set
            for agent in self.__get_agents_not_fulfilled__():
                self.logger.info(f"Agent {agent.name} (id: {agent.id}) is playing")
                alg.set_curr_agent(agent.id)
                partial_seed, new_spreads = alg.run() # Execution of the algorithm
                # partial_seed is an array of length 1, but we decided to make it general, as in the future we may
                # want to select more than one node for every turn
                for node in partial_seed:
                    activate_node(graph=self.graph, node=node, agent=agent)
                # Update the spreads
                for a in self.agents:
                    a.spread = new_spreads[a.name]
                # Update the seed set
                agent.seed.extend(partial_seed)
                self.__register_history__(turn_counter, self.agents) # Register the state of the agents at the end of the turn
                turn_counter += 1
            round_counter += 1
        self.logger.info(f"Game over")
        # Compute the total time
        execution_time = time.time() - start_time
        self.logger.info(f"Seed sets found:")
        for a in self.agents:
            self.logger.info(f"{a.name}: {[self.inverse_mapping[s] for s in a.seed]}")
        self.logger.info(f"Starting the spreads estimation with {self.r} simulation(s)")
        # Do r simulations to estimate the spread of influence of the agents with the seed sets found
        spreads = simulation(graph=self.graph, diff_model=self.diff_model, agents=self.agents, r=self.r, verbose=True)
        # Build the result applying the inverse mapping to re-label the nodes with their original names
        for a in self.agents:
            a.seed = [self.inverse_mapping[s] for s in a.seed]
            a.spread = spreads[a.name]
        seed = {a.name: a.seed for a in self.agents}
        spread = {a.name: a.spread for a in self.agents}
        self.result = {
            'seed': seed,
            'spread': spread,
            'execution_time': execution_time
        }
        return seed, spread, execution_time


# Utility function
def find_hierarchy(superclass):
    """
    This method explores the namespace and recursively builds an array representing all the subclasses.
    There are four super classes: Algorithm, DiffusionModel, InfluenceProbability and EndorsementPolicy. From each of
    these four, there is a hierarchy of subclasses. So if this method is called on DiffusionModel, it will return an array
    with all the subclasses names. If it's called on Algorithm (which has subclasses that also have their own subclasses),
    the result array will only contain the leaf nodes names (which are the ones that can be instantiated and used),
    without the intermediate nodes.
    :param superclass: the superclass which hierarchy has to be explored.
    :return: an array containing all the subclasses names.
    """
    subclasses = []
    for subclass in superclass.__subclasses__():
        # Subclasses who are not leaf nodes don't have the 'name' attribute, so they simply have to be explored
        # without adding their name to the result array
        if hasattr(subclass, 'name'):
            subclasses.append((subclass.name, subclass))
            subclasses.extend(find_hierarchy(subclass))
        else:
            subclasses.extend(find_hierarchy(subclass))
    return subclasses