import sys
import os
import copy
import time
import numpy as np

import dask

import optimize_io
from optimize_io.main import optimize_func

import tests_utils
from tests_utils import *


def test_sum():
    """ Test if the sum of two blocks yields the good
    result usign our optimization function.
    """
    output_dir = os.environ.get('OUTPUT_DIR')
    data_path = get_test_array()
    key = 'data'
    for nb_arr_to_sum in [2, 35, 70, 95]:
        for chunk_shape in ['blocks_dask_interpol']:  # tests_utils.chunk_shapes:
            
            dask.config.set({'optimizations': []})
            result_non_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum).compute()

            dask.config.set({'optimizations': [optimize_func]})
            result_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum).compute()

            assert np.array_equal(result_non_opti, result_opti)
