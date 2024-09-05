import os.path
import networkx as nx

# TODO: add .txt and other extensions
def read_graph(file_path):
    _, ext = os.path.splitext(file_path)
    if ext == '.gml':
        return nx.read_gml(file_path)
    else:
        raise TypeError(f"Graph format {ext} is not supported")

def find_hierarchy(superclass):
    subclasses = [(s.name, s) for s in superclass.__subclasses__()]
    for (name, subclass) in subclasses:
        subclasses.extend(find_hierarchy(subclass))
    return {name: subclass for (name, subclass) in subclasses}

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