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


#----------------------------------------------------------- SPLIT TESTING 

def split(split_filepath, config, nb_blocks):
    # overwrite if split file already exists
    if os.path.isfile(split_filepath):
        os.remove(split_filepath)

    # compute the split
    with h5py.File(split_filepath, 'w') as f:
        # get array parts to be saved in different places
        arr = get_or_create_array(config)
        arr_list = get_arr_list(arr, nb_blocks)  # arr_list == splits to be saved

        datasets = list()
        for i, a in enumerate(arr_list):
            print("creating dataset in split file -> dataset path: ", '/data' + str(i))
            print("storing data of shape", a.shape)
            datasets.append(f.create_dataset('/data' + str(i), shape=a.shape))

        print("storing...")    
        da.store(arr_list, datasets, compute=True)
        print("stored with success.")

    return 


def stored_file(split_filepath):
    print("Checking split file integrity...")
    with h5py.File(split_filepath, 'r') as f:
        print("file", f)
        print("keys", list(f.keys()))
        assert len(list(f.keys())) != 0
    print("Integrity check passed.")


def store_correct(split_filepath, arr_list, logical_chunks_shape):
    print("Testing", len(arr_list), "matches...")
    with h5py.File(split_filepath, 'r') as f:
        for i, a in enumerate(arr_list):
            stored_a = da.from_array(f['/data' + str(i)])
            print("split shape:", stored_a.shape)
            
            stored_a.rechunk(chunks=logical_chunks_shape)
            print("split rechunked to:", stored_a.shape)
            print("will be compared to : ", a.shape)

            print("Testing all close...")
            test = da.allclose(stored_a, a)
            assert test.compute()
            print("Passed.")


def create_arrays_for_comparison(config, nb_blocks):
    arr = get_or_create_array(config)
    arr_list = get_arr_list(arr, nb_blocks) 
    return arr_list
    

def store_test(optimized):
    # setup config
    orig_arr_filepath = os.path.join(os.getenv('DATA_PATH'), 'sample_array_nochunk.hdf5')
    split_filepath = os.path.join(os.getenv('DATA_PATH'), "split_file.hdf5")
    nb_blocks = 2

    for chunks_shape_name in list(CHUNK_SHAPES_EXP1.keys()): # ['slabs_dask_interpol']:
        for scheduler_optimimzed in [True, False]:
            # select chunk shape to use for the split (=split shape)
            chunks_shape = CHUNK_SHAPES_EXP1[chunks_shape_name]

            # configuration -> setup 
            config = CaseConfig(array_filepath=orig_arr_filepath,
                                chunks_shape=chunks_shape)
            config.optimization(opti=optimized, 
                                scheduler_opti=scheduler_optimimzed, 
                                buffer_size=4 * ONE_GIG)
            configure_dask(config, optimize_func)

            # select what you want to do 
            do_store, do_check = (True, True)

            if do_store:
                # split
                split(split_filepath, config, nb_blocks)
                # checker
                stored_file(split_filepath)

            if do_check:
                # test output of split
                arr_list = create_arrays_for_comparison(config, nb_blocks)
                store_correct(split_filepath, arr_list, chunks_shape)


def test_store_optimized():
    store_test(True)


def test_store_non_optimized():
    store_test(False)    