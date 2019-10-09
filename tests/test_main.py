import sys
import os
import copy
import time
import h5py
import numpy as np

import dask
import dask.array as da

import optimize_io
from optimize_io.main import optimize_func

import tests_utils
from tests_utils import *


def test_sum():
    """ Test if the sum of two blocks yields the good
    result usign our optimization function.
    """
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array_nochunk.hdf5')
    output_dir = os.environ.get('OUTPUT_DIR')
    key = 'data'
    for nb_arr_to_sum in [2]:
        for chunk_shape in list(CHUNK_SHAPES_EXP1.keys()): 
            print("chunk shape", chunk_shape)
            # prepare test case
            new_config = CaseConfig(array_filepath=data, chunks_shape=CHUNK_SHAPES_EXP1[chunk_shape])
            new_config.sum_case(nb_chunks=nb_arr_to_sum)

            # run in opti and non opti modes
            dask.config.set({'optimizations': []})
            arr = get_test_arr(new_config)
            dask.config.set({
                'io-optimizer': {
                    'chunk_shape': get_dask_array_chunks_shape(arr),
                    'memory_available': 4 * ONE_GIG
                }
            })
            result_non_opti = arr.compute()

            dask.config.set({'optimizations': [optimize_func]})
            result_opti = get_test_arr(new_config).compute()
            assert np.array_equal(result_non_opti, result_opti)
            print("passed.")

            # viz
            """file_name = chunk_shape + '_' + str(nb_arr_to_sum)"""

            """
            dask.config.set({'optimizations': []})
            arr = get_test_arr(new_config)
            output_path = os.path.join(output_dir, file_name + '_non_opti.png')
            arr.visualize(filename=output_path, optimize_graph=True)"""

            """dask.config.set({'optimizations': [optimize_func]})
            opti_arr = get_test_arr(new_config)
            output_path = os.path.join(output_dir, file_name + '_opti.png')
            opti_arr.visualize(filename=output_path, optimize_graph=True)"""


#TODO: WARNING: add chunk_shape to config!!!
def test_store():
    """ Test the storing procedure with optimization.
    """

    def verify_result(logical_chunks_shape, array_parts, split_file_path):
        print("\n------VERIFICATION STEP------")

        with h5py.File(split_file_path) as f:
            for i, a in enumerate(array_parts):
                stored_a = da.from_array(f['/data' + str(i)])
                stored_a.rechunk(chunks=logical_chunks_shape)
                test = da.allclose(stored_a, a)
                assert test.compute()
            print("passed.")

    def run_store(config):
        """ Main function of this test
        """
        # compute test case array
        print("\n------COMPUTATION STEP------")
        arr = get_test_arr(config)
        arr.compute()  
        
        # case basic array 
        config.test_case = None
        arr_no_case = get_test_arr(config)
        arr_list = get_arr_list(arr_no_case, config.nb_blocks)
        logical_chunks_shape = get_dask_array_chunks_shape(arr_no_case)

        verify_result(logical_chunks_shape, arr_list, config.out_filepath)
        return

    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array_nochunk.hdf5')
    split_file_path = os.path.join(os.getenv('DATA_PATH'), "split_file.hdf5")
    nb_blocks = 2

    for opti in [False, True]:
        for chunk_shape in ['blocks_previous_exp']: 
            if opti:
                for scheduler_opti in [False, True]:
                    print("\n------CONFIGURATION------")
                    print("chunks shape:", chunk_shape)
                    print("optimisation enabled:,", opti)
                    print("scheduler optimisation enabled:", scheduler_opti)
                    print("------------")
                    new_config = CaseConfig(array_filepath=data, 
                                            chunks_shape=CHUNK_SHAPES_EXP1[chunk_shape])
                    new_config.split_case(in_filepath=None, out_filepath=split_file_path, nb_blocks=nb_blocks)
                    new_config.optimization(opti=True, scheduler_opti=scheduler_opti, buffer_size=4 * ONE_GIG)
                    configure_dask(new_config, optimize_func)
                    run_store(new_config)       
            else:
                print("\n------CONFIGURATION------")
                print("chunks shape:", chunk_shape)
                print("optimisation disabled")
                print("scheduler optimisation disabled")
                print("------------")
                new_config = CaseConfig(array_filepath=data, 
                                        chunks_shape=CHUNK_SHAPES_EXP1[chunk_shape])
                new_config.split_case(in_filepath=None, out_filepath=split_file_path, nb_blocks=nb_blocks)
                new_config.optimization(opti=False, scheduler_opti=False, buffer_size=4 * ONE_GIG)
                configure_dask(new_config, optimize_func)
                run_store(new_config)   


if __name__ == "__main__":
    test_store()