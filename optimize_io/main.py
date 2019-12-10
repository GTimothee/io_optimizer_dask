import os
import sys
import math
import time
import optimize_io
from optimize_io.clustered import apply_clustered_strategy
from optimize_io.modifiers import get_used_proxies, get_array_block_dims
from tests_utils import LOG_TIME
import datetime 
import logging

out_dir = os.environ.get('OUTPUT_DIR')
logging.basicConfig(filename=os.path.join(out_dir, LOG_TIME + '.log'), level=logging.DEBUG) # to be set to WARNING


DEBUG_MODE = True

def clustered_optimization(graph):
    """ Applies clustered IO optimization on a Dask graph.

    Arguments:
    ----------
        graph : dark_array.dask.dicts
    """
    print("Finding proxies.")
    chunk_shape, dicts = get_used_proxies(graph)

    print("Launching optimization algorithm.") 
    apply_clustered_strategy(graph, dicts, chunk_shape)
    return graph


def optimize_func(dsk, keys):
    """ Apply an optimization on the dask graph.
    Main function of the library. 

    Arguments:
    ----------
        dsk: dask graph
        keys: 

    Returns: 
    ----------
        the optimized dask graph
    """
    t = time.time()
    dask_graph = dsk.dicts
    dask_graph = clustered_optimization(dask_graph)
    logging.info("Time spent to create the graph: {0:.2f} milliseconds.".format((time.time() - t) * 1000))

    if DEBUG_MODE:
        raise ValueError("stop here")
    return dsk