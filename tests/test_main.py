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


def _sum():
    """ Test if the sum of two blocks yields the good
    result usign our optimization function.
    """
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array_nochunk.hdf5')
    output_dir = os.environ.get('OUTPUT_DIR')
    key = 'data'
    for nb_arr_to_sum in [2]:
        for chunk_shape in ['blocks_previous_exp']: # list(CHUNK_SHAPES_EXP1.keys()): 
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
def store():
    """ Test the storing procedure with optimization.
    """

    def store(file_name, array_parts):
        # remove split file if already exists
        file_path = os.path.join(file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

        with h5py.File(file_path, 'w') as f:
            dsets = list()
            for i, a in enumerate(array_parts):
                d = f.create_dataset('/data' + str(i), shape=a.shape)
                dsets.append(d)

            s = da.store(array_parts, dsets, compute=False)
            s.compute()


    def verify_result(array_parts, split_file_path):
        new_config.opti = False
        configure_dask(new_config, optimize_func)
        with h5py.File(split_file_path) as f:
            for i, a in enumerate(array_parts):
                stored_a = da.from_array(f['/data' + str(i)])
                stored_a.rechunk(chunks=(220, 242, 200))
                test = da.allclose(stored_a, a)
                assert test.compute()


    def run_store(new_config):
        """ Main function of this test
        """
        # get splits to be stored
        arr = get_test_arr(new_config)
        a1 = arr[:220,:484,:400]
        a2 = arr[:220,:484,400:800]

        # set optimization or not
        configure_dask(new_config, optimize_func)
        dask.config.set({
            'io-optimizer': {
                'chunk_shape': get_dask_array_chunks_shape(arr)
            }
        })

        # run test
        split_file_path = os.path.join(os.getenv('DATA_PATH'), "file1.hdf5")
        store(split_file_path, [a1, a2])

        a1 = arr[:220,:484,:400]
        a2 = arr[:220,:484,400:800]
        verify_result([a1, a2], split_file_path)

    for opti in [False, True]:
        for chunk_shape in first_exp_shapes.keys(): 
            data = os.path.join(os.getenv('DATA_PATH'), 'sample_array_nochunk.hdf5')
            new_config = CaseConfig(opti=opti,
                                    scheduler_opti=None, 
                                    out_path=None,
                                    buffer_size=5 * ONE_GIG, 
                                    input_file_path=data, 
                                    chunk_shape=first_exp_shapes[chunk_shape])
            new_config.create_or_overwrite(None, SUB_BIGBRAIN_SHAPE, overwrite=False)
            run_store(new_config)       


if __name__ == "__main__":
    _sum()