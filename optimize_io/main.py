import sys
import math
import optimize_io
from  optimize_io.clustered import *
from  optimize_io.get_slices import *
from  optimize_io.get_dicts import *


__all__ = ("clustered_optimization",
           "convert_slices_list_to_numeric_slices")


def clustered_optimization(graph):
    """ Main function of the library. Applies clustered IO optimization on a Dask graph.
    The first functions calls are used to extract information needed for the clustered optimization.
    Among other things the following association tables are needed for the processing:
        slices_dict :                       proxy-array -> list of slices
        array_to_original :                 proxy-array -> original-array
        original_array_chunks :             original-array-name -> block shapes (anciennement dims_dict) [shapes of each logical block]
        original_array_shapes :             original-array-name -> original-array shape
        original_array_blocks_shape :       original-array-name -> nb blocks in each dim
        deps_dict :                         proxy-array -> proxy-array-dependent tasks list
    args:
        graph : dask_array.dask.dicts
    """
    # collect information
    used_getitems = get_used_getitems_from_graph(graph, undirected=False)
    slices_dict, deps_dict = get_slices_from_dask_graph(graph, used_getitems)
    array_to_original, original_array_chunks, original_array_shapes, original_array_blocks_shape = get_arrays_dictionaries(graph, slices_dict)
    slices_dict = convert_slices_list_to_numeric_slices(slices_dict, array_to_original, original_array_blocks_shape)

    # apply optimization
    graph = apply_clustered_strategy(graph, slices_dict, deps_dict, array_to_original, original_array_chunks, original_array_shapes, original_array_blocks_shape)
    return graph


def convert_slices_list_to_numeric_slices(slices_dict, array_to_original, original_array_blocks_shape):
    """ Converts a list of 3d position of blocks in the source array into their numeric position.
    Ex: [(0,0,0), (0,0,1)] become [0, 1]
    args:
        slices_dict:                    proxy-array -> list of slices
        array_to_original:              proxy-array -> original-array
        original_array_blocks_shape:    original-array-name -> nb blocks in each dim
    """
    for proxy_array, slices_list in slices_dict.items():
        slices_list = sorted(list(set(slices_list)))
        img_nb_blocks_per_dim = original_array_blocks_shape[array_to_original[proxy_array]]
        slices_list = [_3d_to_numeric_pos(s, img_nb_blocks_per_dim, order='C') for s in slices_list]
        slices_dict[proxy_array] = slices_list
    return slices_dict
