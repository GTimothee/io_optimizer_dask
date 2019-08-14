import sys
import os
import copy
import time
import numpy as np

import dask
from dask.array.io_optimization import optimize_func
import dask_utils_perso


import optimize_io
from optimize_io.main import *


import tests_utils 
from tests_utils import *



def test_convert_slices_list_to_numeric_slices():
    """ Function to test convert_slices_list_to_numeric_slices.
    """
    proxy_array_name = 'array-6f8'
    original_array_name = "array-original-645"
    array_to_original = {proxy_array_name: original_array_name}
    original_array_chunks = {original_array_name: (10, 20, 30)}
    original_array_blocks_shape = {original_array_name: (5, 3, 2)}
    slices_dict = {'array-6f8': [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (0,2,0), (0,2,1), (1,0,0)]}
    slices_dict = convert_slices_list_to_numeric_slices(slices_dict, array_to_original, original_array_blocks_shape)
    expected = [0,1,2,3,4,5,6]
    assert sorted(expected) == slices_dict[proxy_array_name]


def test_sum():
    """ Test if the sum of two blocks yields the good result usign our optimization function.
    """
    output_dir = os.environ.get('OUTPUT_DIR')
    data_path = get_test_array()
    key = 'data'
    nb_arr_to_sum = 2
    for chunk_shape in tests_utils.chunk_shapes:
        # non opti
        arr = get_dask_array_from_hdf5(data_path, key)
        dask_array = add_chunks(arr, chunk_shape, number_of_arrays=nb_arr_to_sum)
        result_non_opti = dask_array.sum()

        # viz
        """
        output_path = os.path.join(output_dir, 'test_non_opti' + chunk_shape + '.png')
        result_non_opti.visualize(filename=output_path, optimize_graph=True)
        """

        # opti
        dask.config.set({'optimizations': [optimize_func]})
        arr = get_dask_array_from_hdf5(data_path, key)
        dask_array = add_chunks(arr, chunk_shape, number_of_arrays=nb_arr_to_sum)
        result_opti = dask_array.sum()

        # viz
        """
        output_path = os.path.join(output_dir, 'test_opti' + chunk_shape + '.png')
        result_opti.visualize(filename=output_path, optimize_graph=True)
        """

        assert np.array_equal(result_opti.compute(), result_non_opti.compute())


def test_adding_chunks(computation=True, visuals=True, run_non_optimized=True, run_optimized=True):

    def do_a_run(number_of_arrays, viz, prefix=None, suffix=None):
        results = list()
        for chunk_shape in tests_utils.chunk_shapes:
            # free cache for independence between the runs
            os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')

            # load array
            data_path = get_test_array()
            key = "data"
            arr = get_dask_array_from_hdf5(data_path, key)
            dask_array = add_chunks(arr, chunk_shape, number_of_arrays=number_of_arrays)

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
    _ = do_a_run(number_of_arrays, viz=True, prefix='./output_imgs/', suffix='non-opti.png')   
    results = do_a_run(number_of_arrays, viz=False)             

    # with opti
    # set optimization function
    dask.config.set({'optimizations': [optimize_func]})
    dask.config.set({'io-optimizer': {'memory_available':ONE_GIG}})
    print("with optimization")
    _ = do_a_run(number_of_arrays, viz=True, prefix='./output_imgs/', suffix='opti.png')
    results_opti = do_a_run(number_of_arrays, viz=False)
    if run_non_optimized:  # if both have been ran, we can compare their results
        for result_non_opti, result_opti in zip(results, results_opti):
            assert np.array_equal(result_non_opti, result_opti)
