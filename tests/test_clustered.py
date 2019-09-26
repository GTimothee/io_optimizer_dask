import os 

from optimize_io.clustered import *
from optimize_io.modifiers import get_used_proxies

from tests_utils import get_test_arr, CaseConfig, ONE_GIG, neat_print_graph, get_arr_shapes, SUB_BIGBRAIN_SHAPE

import sys


def get_case_1():
    # case 1 : continous blocks
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
    config = CaseConfig(data, None)
    arr = get_test_arr(config)

    shape, chunks, blocks_dims = get_arr_shapes(arr)
    _3d_pos = numeric_to_3d_pos(34, blocks_dims, 'C')
    dims = [(_3d_pos[0]+1) * chunks[0],
            (_3d_pos[1]+1) * chunks[1],
            (_3d_pos[2]+1) * chunks[2]]
    arr = arr[0:dims[0], 0:dims[1], 0:dims[2]]
    arr = arr + 1
    return arr


def test_get_covered_blocks():
    """
    remainder chunk shape: 220 242 200
    """
    slice_tuple = (slice(0, 225, None), slice(242, 484, None), slice(500, 700, None))
    chunk_shape = (220, 242, 200)
    ranges = get_covered_blocks(slice_tuple, chunk_shape)
    assert [list(r) for r in ranges] == [[0, 1], [1], [2, 3]]

    slice_tuple = (slice(0, 220, None), slice(0, 242, None), slice(0, 200, None))
    x_range, y_range, z_range = get_covered_blocks(slice_tuple, chunk_shape)
    ranges = get_covered_blocks(slice_tuple, chunk_shape)
    assert [list(r) for r in ranges] == [[0], [0], [0]]


def test_get_blocks_used():
    # case 1 : continous blocks
    arr = get_case_1()

    # routine to get the needed data
    # we assume those functions have been tested before get_blocks_used
    dicts = get_used_proxies(arr.dask.dicts)

    origarr_name = list(dicts['origarr_to_obj'].keys())[0]
    arr_obj = dicts['origarr_to_obj'][origarr_name]
    strategy, max_blocks_per_load = get_load_strategy(ONE_GIG, 
                                                      arr_obj.shape, 
                                                      arr_obj.chunks)

    # actual test of the function
    blocks_used, block_to_proxies = get_blocks_used(dicts, origarr_name, arr_obj)
    expected = list(range(35))
    assert blocks_used == expected


def test_create_buffers():
    """
    row size = 7 blocks
    slices size = 35 blocks

    default mem = 1 000 000 000

    # (220 * 242 * 200) = 10 648 000 
    # with 4 bytes per pixel, we have maximum 23 blocks that can be loaded
    # 21 blocks contiguous and not overlaping then the last 14 blocks
    => strategy: buffer=row_size max 
    """

    # case 1 : continous blocks
    arr = get_case_1()

    dicts = get_used_proxies(arr.dask.dicts)
    origarr_name = list(dicts['origarr_to_obj'].keys())[0]
    buffers = create_buffers(origarr_name, dicts)
    
    expected = [list(range(21)), list(range(21, 35))]
    
    print("buffers - ", buffers)
    print("expected - ", expected)
    assert buffers == expected


def test_create_buffer_node():
    # preparation
    arr = get_case_1()
    graph = arr.dask.dicts
    dicts = get_used_proxies(graph)
    origarr_name = list(dicts['origarr_to_obj'].keys())[0]
    buffers = create_buffers(origarr_name, dicts)
        
    # apply function
    keys = list()
    for buffer in buffers:            
        key = create_buffer_node(graph, origarr_name, dicts, buffer)    
        keys.append(key)

    # test output
    buffers_key = origarr_name.split('-')[-1] + '-merged'

    indices = set()
    for buffer_key in graph[buffers_key].keys():
        _, start, end = buffer_key
        indices.add((start, end))

    buffers = set([(b[0], b[-1]) for b in buffers])

    assert buffers_key in graph.keys()
    assert len(indices) == len(buffers)
    assert buffers == indices

    
