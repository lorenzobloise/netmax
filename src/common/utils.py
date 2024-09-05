import networkx as nx

def read_graph(input_path):
    """
    :return: The directed graph in networkx format
    :param input_path: Path of the dataset (.txt file)
    """
    graph = nx.DiGraph()
    with open(input_path, 'r') as f:
        data_lines = f.readlines()
    node_num = int(data_lines[0].split()[0])
    nodes = [i for i in range(node_num)]
    graph.add_nodes_from(nodes)
    for data_line in data_lines[1:]:
        start, end, weight = data_line.split()
        graph.add_edge(int(start), int(end), p=float(weight))
    return graph

def __my_similarity__(list1, list2):
    number_of_elements_of_list1_in_list2 = 0
    for x in list1:
        if x in list2:
            number_of_elements_of_list1_in_list2 += 1
    return number_of_elements_of_list1_in_list2 / len(list1)

def __calculate_similarities__(dataframe):
    # Calculate similarities between seed sets of different algorithms
    algorithms = dataframe["algorithm"].unique()
    for i, row in dataframe.iterrows():
        current_seed_set = row["seed"]
        for elem in algorithms:
            other_seed_set=dataframe[dataframe["algorithm"] == elem].iloc[0]["seed"]
            dataframe.loc[i, f'similarity_{elem}'] = __my_similarity__(current_seed_set, other_seed_set)
    return dataframe

def find_hierarchy(superclass):
    subclasses = []
    for subclass in superclass.__subclasses__():
        if hasattr(subclass, 'name'):
            subclasses.append((subclass.name, subclass))
        else:
            subclasses.extend(find_hierarchy(subclass))

    return subclasses