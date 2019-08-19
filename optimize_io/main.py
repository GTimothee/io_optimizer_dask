import sys
import math
import optimize_io
from optimize_io.clustered import *
from optimize_io.get_slices import *
from optimize_io.get_dicts import *
from optimize_io.modifiers import origarr_to_used_proxies_dict


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
    (origarr_to_slices_dict, 
        origarr_to_used_proxies_dict,
        original_array_shapes,
        original_array_chunks,
        original_array_blocks_shape) = get_used_proxies(graph, undirected=False)
            
    origarr_to_slices_dict = convert_slices_list_to_numeric_slices(
        origarr_to_slices_dict, array_to_original, original_array_blocks_shape)
        
    print("****slices:\n", slices_dict)

    # apply optimization
    graph = apply_clustered_strategy(
        graph,
        slices_dict,
        deps_dict,
        array_to_original,
        original_array_chunks,
        original_array_shapes,
        original_array_blocks_shape)

    for k, v in graph.items():
        print("\nkey", k)
        print(v, "\n")
    return graph


def convert_slices_list_to_numeric_slices(
        slices_dict,
        array_to_original,
        original_array_blocks_shape):
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
        slices_list = [
            _3d_to_numeric_pos(
                s,
                img_nb_blocks_per_dim,
                order='C') for s in slices_list]
        slices_dict[proxy_array] = slices_list
    return slices_dict
