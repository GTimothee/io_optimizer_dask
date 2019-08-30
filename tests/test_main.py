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
    for nb_arr_to_sum in [35]:
        for chunk_shape in tests_utils.chunk_shapes: 
            
            dask.config.set({'optimizations': []})
            result_non_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum).compute()

            dask.config.set({'optimizations': [optimize_func]})
            result_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum).compute()

            assert np.array_equal(result_non_opti, result_opti)

            # viz
            file_name = chunk_shape + '_' + str(nb_arr_to_sum)
            dask.config.set({'optimizations': []})
            result_non_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
            output_path = os.path.join(output_dir, file_name + '_non_opti.png')
            result_non_opti.visualize(filename=output_path, optimize_graph=True)

            dask.config.set({'optimizations': [optimize_func]})
            result_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
            output_path = os.path.join(output_dir, file_name + '_opti.png')
            result_opti.visualize(filename=output_path, optimize_graph=True)
