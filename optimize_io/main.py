import sys
import math
import optimize_io
from optimize_io.clustered import apply_clustered_strategy
from optimize_io.modifiers import get_used_proxies


def clustered_optimization(graph):
    """ Main function of the library. Applies clustered IO optimization on a Dask graph.
    """
    dicts = get_used_proxies(graph) 
    apply_clustered_strategy(graph, dicts)
    return graph