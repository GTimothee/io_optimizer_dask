
from optimize_io.clustered import *
from optimize_io.modifiers import get_used_proxies

from tests_utils import get_test_arr, get_arr_shapes, ONE_GIG

import sys


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
    arr = get_test_arr(case=None)
    shape, chunks, blocks_dims = get_arr_shapes(arr)
    _3d_pos = numeric_to_3d_pos(34, blocks_dims, 'C')
    dims = [(_3d_pos[0]+1) * chunks[0],
            (_3d_pos[1]+1) * chunks[1],
            (_3d_pos[2]+1) * chunks[2]]
    arr = arr[0:dims[0], 0:dims[1], 0:dims[2]]
    arr = arr + 1

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





