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