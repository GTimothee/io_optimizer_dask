import os 

from optimize_io.clustered import *
from optimize_io.modifiers import get_used_proxies

from tests_utils import *

import sys


# TODO: make tests with different chunk shapes

def get_case_1():
    # case 1 : continous blocks
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array_nochunk.hdf5')
    config = CaseConfig(data, (770, 605, 700))
    arr = get_test_arr(config)
    cs = get_dask_array_chunks_shape(arr)
    dask.config.set({
        'io-optimizer': {
            'chunk_shape': (220, 240, 200),
            'memory_available': 4 * ONE_GIG
        }
    })

    # attention à l'énoncé ci-dessous # TODO: replace by sum_case
    shape, chunks, blocks_dims = get_arr_shapes(arr)
    _3d_pos = numeric_to_3d_pos(5, blocks_dims, 'C')
    dims = [(_3d_pos[0]+1) * chunks[0],
            (_3d_pos[1]+1) * chunks[1],
            (_3d_pos[2]+1) * chunks[2]]
    arr = arr[0:dims[0], 0:dims[1], 0:dims[2]]
    arr = arr + 1
    
    return arr, config


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
    arr, config = get_case_1()

    # routine to get the needed data
    # we assume those functions have been tested before get_blocks_used
    _, dicts = get_used_proxies(arr.dask.dicts)

    origarr_name = list(dicts['origarr_to_obj'].keys())[0]
    arr_obj = dicts['origarr_to_obj'][origarr_name]
    strategy, max_blocks_per_load = get_load_strategy(4 * ONE_GIG, 
                                                      (770, 605, 700), 
                                                      config.chunks_shape) 

    # actual test of the function
    blocks_used, block_to_proxies = get_blocks_used(dicts, origarr_name, arr_obj, config.chunks_shape)
    expected = [0,1,4,5]
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
    arr, config = get_case_1()

    _, dicts = get_used_proxies(arr.dask.dicts)
    origarr_name = list(dicts['origarr_to_obj'].keys())[0]
    buffers = create_buffers(origarr_name, dicts, config.chunks_shape)
    
    expected = [[0,1], [4,5]]
    
    print("buffers - ", buffers)
    print("expected - ", expected)
    assert buffers == expected


def test_create_buffer_node():
    # preparation
    arr, config = get_case_1()
    graph = arr.dask.dicts
    _, dicts = get_used_proxies(graph)
    origarr_name = list(dicts['origarr_to_obj'].keys())[0]
    buffers = create_buffers(origarr_name, dicts, config.chunks_shape)
        
    # apply function
    keys = list()
    for buffer in buffers:            
        key = create_buffer_node(graph, origarr_name, dicts, buffer, config.chunks_shape)    
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

    
def test_get_load_strategy():
    strat, max_nb_blocks = get_load_strategy(4000, (10, 10, 1), None, nb=4)
    assert strat == 'blocks'
    assert max_nb_blocks == 10


def test_start_new_buffer():    
    # WARNING: only blocks strategy have been implemented so far

    strategy = "blocks"
    nb_blocks_per_row = 4
    max_nb_blocks_per_buffer = 10

    test_buffers = [[0,1,2,3,4,5,6,7,8,9],
                    [0,1,2,3],
                    [1,2],
                    [1,2]]

    blocks = [10,
              4,
              3,
              4] 

    expected = [True,
                True,
                False,
                True]

    i = 1
    for curr_buffer, b, e in zip(test_buffers, blocks, expected):
        print("case", i)
        
        res = start_new_buffer(curr_buffer, 
                            b, 
                            curr_buffer[-1], 
                            strategy, 
                            nb_blocks_per_row, 
                            max_nb_blocks_per_buffer)

        assert res == e
        i += 1


def test_overlap_slices():
    curr_buffers = [
        list(range(16)),
        list(range(8)),
        list(range(20, 24))
    ]
    buffers = [
        list(range(16, 20)),
        list(range(8, 12)),
        list(range(56, 60))
    ]
    expected = [
        True,
        False,
        True
    ]
    blocks_shape = (4, 4, 4)

    i = 1
    for curr_buff, buff, exp in zip(curr_buffers, buffers, expected):
        print("case", i)
        out = overlap_slice(curr_buff, buff, blocks_shape)
        i += 1
        assert out == exp


def test_merge_rows():
    blocks_shape = (4, 4, 4)
    nb_blocks_per_row = 4
    max_blocks_per_load = 21

    buffers_list = [
        [list(range(x*4, x*4 +4)) for x in range(8)]
    ]
    expected = [
        [list(range(16)), list(range(16, 32))]
    ]
    
    for buffers, exp in zip(buffers_list, expected):
        out = merge_rows(buffers, blocks_shape, nb_blocks_per_row, max_blocks_per_load)
        assert out == exp


def test_merge_slices():
    nb_blocks_per_slice = 16
    max_blocks_per_load = 35
    buffers_list = [
        [list(range(x * 16, x * 16 + 16)) for x in range(4)]
    ]
    expected = [
        [list(range(32)), list(range(32, 64))]
    ]

    for buffers, exp in zip(buffers_list, expected):    
        out = merge_slices(buffers, nb_blocks_per_slice, max_blocks_per_load)
        assert out == exp


def test_buffering():
    strategy = "blocks"
    blocks_shape = (4, 4, 4)
    max_nb_blocks_per_buffer = 9

    blocks = [0,1,2,3,6,7,8,9,10,11,12]
    row_concat, slices_concat = (False, False)
    exp = [[0,1,2,3], [6,7], [8, 9, 10, 11], [12]]
    out = buffering(blocks, strategy, blocks_shape, max_nb_blocks_per_buffer, row_concat=row_concat, slices_concat=slices_concat)
    assert out == exp

if __name__ == "__main__":
    test_create_buffers()