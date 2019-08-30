import sys
import math
import time
import optimize_io
from optimize_io.clustered import apply_clustered_strategy
from optimize_io.modifiers import get_used_proxies
from tests_utils import neat_print_graph


def clustered_optimization(graph):
    """ Main function of the library. Applies clustered IO optimization on a Dask graph.
    """
    print("getting proxies")
    dicts = get_used_proxies(graph)
    print("applying strategy") 
    apply_clustered_strategy(graph, dicts)
    # neat_print_graph(graph)
    return graph


def optimize_func(dsk, keys):
    t = time.time()
    dask_graph = dsk.dicts
    dask_graph = clustered_optimization(dask_graph)
    t = time.time() - t
    print("time to create graph:", t)
    return dsk