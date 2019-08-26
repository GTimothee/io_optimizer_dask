
from optimize_io.clustered import *
from optimize_io.modifiers import get_used_proxies

from tests_utils import get_test_arr, get_arr_shapes

import sys


def test_get_covered_blocks():
    """
    remainder chunk shape: 220 242 200
    """
    slice_tuple = (slice(0, 225, None), slice(242, 484, None), slice(500, 700, None))
    chunk_shape = (220, 242, 200)
    x_range, y_range, z_range = get_covered_blocks(slice_tuple, chunk_shape)
    assert list(x_range) == [0, 1]
    assert list(y_range) == [1]
    assert list(z_range) == [2, 3]


    





