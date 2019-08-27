import sys
import os
import copy
import time
import numpy as np

import dask
from dask.array.io_optimization import optimize_func
import dask_utils_perso


import optimize_io
from optimize_io.main import clustered_optimization


import tests_utils
from tests_utils import *


def test_sum():
    """ Test if the sum of two blocks yields the good
    result usign our optimization function.
    """
    output_dir = os.environ.get('OUTPUT_DIR')
    data_path = get_test_array()
    key = 'data'
    nb_arr_to_sum = 2
    for chunk_shape in tests_utils.chunk_shapes:

        # test results
        dask.config.set({'optimizations': []})
        result_non_opti = get_test_arr(case='sum', nb_arr=2).compute()

        dask.config.set({'optimizations': [optimize_func]})
        result_opti = get_test_arr(case='sum', nb_arr=2).compute()
        
        assert np.array_equal(result_non_opti, result_opti)

        # viz
        dask.config.set({'optimizations': []})
        result_non_opti = get_test_arr(case='sum', nb_arr=2)
        output_path = os.path.join(output_dir, 'test_non_opti' + chunk_shape + '.png')
        result_non_opti.visualize(filename=output_path, optimize_graph=True)

        dask.config.set({'optimizations': [optimize_func]})
        result_opti = get_test_arr(case='sum', nb_arr=2)
        output_path = os.path.join(output_dir, 'test_opti' + chunk_shape + '.png')
        result_opti.visualize(filename=output_path, optimize_graph=True)


"""def test_adding_chunks(
        computation=True,
        visuals=True,
        run_non_optimized=True,
        run_optimized=True):

    def do_a_run(number_of_arrays, viz, prefix=None, suffix=None):
        results = list()
        for chunk_shape in tests_utils.chunk_shapes:
            # free cache for independence between the runs
            os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')

            # load array
            data_path = get_test_array()
            key = "data"
            arr = get_dask_array_from_hdf5(data_path, key)
            dask_array = add_chunks(
                arr, chunk_shape, number_of_arrays=number_of_arrays)

            # process
            t = time.time()
            if not viz:
                r = dask_array.compute()
                results.append(r)
            else:
                if not suffix:
                    print("please give a filename")
                    sys.exit()
                filename = ''.join([prefix, chunk_shape, '-', suffix])
                dask_array.visualize(filename=filename, optimize_graph=True)
            t = time.time() - t

            print("total processing time:", t)

        if viz:
            results = None
        return results

    from multiprocessing.pool import ThreadPool
    dask.config.set(pool=ThreadPool(1))
    number_of_arrays = 2

    # without optidask.config.set({'optimizations': []})
    print("without optimization")
    _ = do_a_run(
        number_of_arrays,
        viz=True,
        prefix='./output_imgs/',
        suffix='non-opti.png')
    results = do_a_run(number_of_arrays, viz=False)

    # with opti
    # set optimization function
    dask.config.set({'optimizations': [optimize_func]})
    dask.config.set({'io-optimizer': {'memory_available': ONE_GIG}})
    print("with optimization")
    _ = do_a_run(
        number_of_arrays,
        viz=True,
        prefix='./output_imgs/',
        suffix='opti.png')
    results_opti = do_a_run(number_of_arrays, viz=False)

    # if both have been ran, we can compare their results
    if run_non_optimized:
        for result_non_opti, result_opti in zip(results, results_opti):
            assert np.array_equal(result_non_opti, result_opti)"""
