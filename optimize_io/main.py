
import sys
import math 
sys.path.insert(0,'/home/user/Documents/workspace/projects/dask') # custom dask
sys.path.insert(1,'/home/user/Documents/workspace/projects/samActivities/tests') 
sys.path.insert(2,'/home/user/Documents/workspace/projects/samActivities/tests/optimize_io')
sys.path.insert(3,'/home/user/Documents/workspace/projects/samActivities/experience3')

import clustered
from clustered import *
from get_slices import *
from get_dicts import *

__all__ = ("main", "convert_slices_list_to_numeric_slices", "get_slice_from_merged_task_name")


def main(graph):
    """
        slices_dict :                       proxy-array -> list of slices
        array_to_original :                 proxy-array -> original-array
        original_array_chunks :             original-array-name -> block shapes (anciennement dims_dict) [shapes of each logical block]
        original_array_shapes :             original-array-name -> original-array shape
        original_array_blocks_shape :       original-array-name -> nb blocks in each dim
        deps_dict :                         proxy-array -> proxy-array-dependent tasks list
    """
    used_getitems = get_used_getitems_from_graph(graph, undirected=False)

    print("used_getitems", len(used_getitems))

    slices_dict, deps_dict = get_slices_from_dask_graph(graph, used_getitems)
    array_to_original, original_array_chunks, original_array_shapes, original_array_blocks_shape = get_arrays_dictionaries(graph, slices_dict)
    slices_dict = convert_slices_list_to_numeric_slices(slices_dict, array_to_original, original_array_blocks_shape)
    graph = apply_clustered_strategy(graph, slices_dict, deps_dict, array_to_original, original_array_chunks, original_array_shapes, original_array_blocks_shape)
    return graph


def convert_slices_list_to_numeric_slices(slices_dict, array_to_original, original_array_blocks_shape):
    for proxy_array, slices_list in slices_dict.items():
        slices_list = sorted(list(set(slices_list)))

        img_nb_blocks_per_dim = original_array_blocks_shape[array_to_original[proxy_array]]
        slices_list = [_3d_to_numeric_pos(s, img_nb_blocks_per_dim, order='C') for s in slices_list]
        slices_dict[proxy_array] = slices_list
    return slices_dict


def get_slice_from_merged_task_name(merged_task_name, target_slice, img_nb_blocks_per_dim):
    """ not used for the moment
    """
    _, _, start_of_block, _ = merged_task_name.split('-')
    
    sot = [s.start for s in target_slice]
    sot = _3d_to_numeric_pos(sot, img_nb_blocks_per_dim, order='C')
    sot = sot - start_of_block

    eot = [s.stop + 1 for s in target_slice]
    eot = _3d_to_numeric_pos(eot, img_nb_blocks_per_dim, order='C')
    eot = eot - start_of_block

    return (slice(sot[0], eot[0], None), 
            slice(sot[1], eot[1], None), 
            slice(sot[2], eot[2], None))