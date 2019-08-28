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
    nb_arr_to_sum = 35
    for chunk_shape in ['blocks_dask_interpol']:  # tests_utils.chunk_shapes:

        # test results
        dask.config.set({'optimizations': []})
        t = time.time()
        result_non_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum).compute()
        t = time.time() - t

        dask.config.set({'optimizations': [optimize_func]})
        t2 = time.time()
        result_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum).compute()
        t2 = time.time() - t2
        
        import csv
        with open(os.path.join(output_dir, 'speeds.csv'), mode='w+') as csv_out:
            writer = csv.writer(csv_out, delimiter=',')
            writer.writerow(['non optimized', t])
            writer.writerow(['optimized', t2])

        assert np.array_equal(result_non_opti, result_opti)

        # viz
        dask.config.set({'optimizations': []})
        result_non_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
        output_path = os.path.join(output_dir, 'test_non_opti' + chunk_shape + '.png')
        result_non_opti.visualize(filename=output_path, optimize_graph=True)

        dask.config.set({'optimizations': [optimize_func]})
        result_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
        output_path = os.path.join(output_dir, 'test_opti' + chunk_shape + '.png')
        result_opti.visualize(filename=output_path, optimize_graph=True)
