import sys
import math
import optimize_io
from optimize_io.clustered import apply_clustered_strategy
from optimize_io.modifiers import get_used_proxies


def clustered_optimization(graph):
    """ Main function of the library. Applies clustered IO optimization on a Dask graph.
    """
    dicts = get_used_proxies(graph, undirected=False) 
    apply_clustered_strategy(graph, dicts)
    neat_print_graph(graph)
    return graph


def neat_print_graph(graph):
    for k, v in graph.items():
        print("\nkey", k)
        print(v, "\n")