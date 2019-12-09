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


def clustered_optimization(graph):
    """ Main function of the library. Applies clustered IO optimization on a Dask graph.
    graph : dark_array.dask.dicts
    """
    print("Finding proxies.")
    chunk_shape, dicts = get_used_proxies(graph)

    print("Launching optimization algorithm.") 
    apply_clustered_strategy(graph, dicts, chunk_shape)
    return graph


def optimize_func(dsk, keys):
    t = time.time()
    dask_graph = dsk.dicts
    dask_graph = clustered_optimization(dask_graph)
    logging.info("Time spent to create the graph: {0:.2f} milliseconds.".format((time.time() - t) * 1000))
    raise ValueError("stop here")
    return dsk