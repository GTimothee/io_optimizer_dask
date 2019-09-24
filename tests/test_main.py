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

import test_modifiers
from test_modifiers import *


def test_sum():
    """ Test if the sum of two blocks yields the good
    result usign our optimization function.
    """
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
    output_dir = os.environ.get('OUTPUT_DIR')
    key = 'data'
    for nb_arr_to_sum in [5]:
        for chunk_shape in first_exp_shapes.keys(): 
            
            # prepare test case
            new_config = CaseConfig(opti=None, 
                             scheduler_opti=None, 
                             out_path=None, 
                             buffer_size=ONE_GIG, 
                             input_file_path=data, 
                             chunk_shape=first_exp_shapes[chunk_shape])
            new_config.create_or_overwrite(None, SUB_BIGBRAIN_SHAPE, overwrite=False)
            new_config.sum_case(nb_chunks=nb_arr_to_sum)

            # run in opti and non opti modes
            dask.config.set({'optimizations': []})
            result_non_opti = get_test_arr(new_config).compute()

            dask.config.set({'optimizations': [optimize_func]})
            result_opti = get_test_arr(new_config).compute()
            assert np.array_equal(result_non_opti, result_opti)

            # viz
            file_name = chunk_shape + '_' + str(nb_arr_to_sum)

            """
            dask.config.set({'optimizations': []})
            arr = get_test_arr(new_config)
            output_path = os.path.join(output_dir, file_name + '_non_opti.png')
            arr.visualize(filename=output_path, optimize_graph=True)"""

            dask.config.set({'optimizations': [optimize_func]})
            opti_arr = get_test_arr(new_config)
            output_path = os.path.join(output_dir, file_name + '_opti.png')
            opti_arr.visualize(filename=output_path, optimize_graph=True)


def test_store():
    """ Test if the processing of the store dask graph works. 
    """
    def get_datasets(file_name, a1, a2):
        file_path = os.path.join(file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        f = h5py.File(file_path, 'w')
        dset1 = f.create_dataset('/data1', shape=a1.shape)
        dset2 = f.create_dataset('/data2', shape=a2.shape)
        return f, dset1, dset2

    output_dir = os.environ.get('OUTPUT_DIR')
    file_name = "store" 

    # ------ non optimized
    # prepare test case
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
    new_config = CaseConfig(opti=None, 
                        scheduler_opti=None, 
                        out_path=None, 
                        buffer_size=ONE_GIG, 
                        input_file_path=data, 
                        chunk_shape=None)
    new_config.create_or_overwrite(None, SUB_BIGBRAIN_SHAPE, overwrite=False)

    # ------ optimized
    arr = get_test_arr(new_config)
    a1 = arr[:220,:484,:400]
    a2 = arr[:220,:484,400:800]

    buffer_size = ONE_GIG
    dask.config.set({'optimizations': [optimize_func]})
    dask.config.set({'io-optimizer': {'memory_available': buffer_size,
                                        'scheduler_opti': True}})
    f, dset1, dset2 = get_datasets("data/file2.hdf5", a1, a2)
    s = da.store([a1, a2], [dset1, dset2], compute=False)
    s.visualize(filename='tests/outputs/img.png', optimize_graph=False)

    # step by step of our optimization 
    dask_graph = s.dask.dicts 
    graph = get_graph_from_dask(dask_graph, undirected=False)  # we want a directed graph

    with open('tests/outputs/remade_graph.txt', "w+") as f:
        for k, v in graph.items():
            f.write("\n\n" + str(k))
            f.write("\n" + str(v))

    root_nodes = get_unused_keys(graph)
    print('\nRoot nodes:')
    for root in root_nodes:
        print(root)

    f.close()


def test_store_non_opti():
    """ Test the storing procedure with optimization.
    """

    def get_datasets(file_name, a1, a2):
        file_path = os.path.join(file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        f = h5py.File(file_path, 'w')
        dset1 = f.create_dataset('/data1', shape=a1.shape)
        dset2 = f.create_dataset('/data2', shape=a2.shape)
        return f, dset1, dset2

    output_dir = os.environ.get('OUTPUT_DIR')
    viz_file_name = "store" 

   
    # prepare test case
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
    new_config = CaseConfig(opti=None, 
                        scheduler_opti=None, 
                        out_path=None, 
                        buffer_size=5 * ONE_GIG, 
                        input_file_path=data, 
                        chunk_shape=None)
    new_config.create_or_overwrite(None, SUB_BIGBRAIN_SHAPE, overwrite=False)

    # ------ optimized
    arr = get_test_arr(new_config)
    a1 = arr[:220,:484,:400]
    a2 = arr[:220,:484,400:800]

    buffer_size = 5 * ONE_GIG

    path = os.path.join(os.getenv('DATA_PATH'), "file1.hdf5")
    f, dset1, dset2 = get_datasets(path, a1, a2)
    s = da.store([a1, a2], [dset1, dset2], compute=False)
    s.compute()
    output_path = os.path.join(output_dir, viz_file_name + '_opti.png')
    # s.visualize(filename=output_path, optimize_graph=True)
    f.close()

    # verification
    a1 = arr[:220,:484,:400]
    a2 = arr[:220,:484,400:800]
    with h5py.File(path) as f:
        file_a1 = da.from_array(f['/data1'])
        file_a2 = da.from_array(f['/data2'])

        file_a1.rechunk(chunks=(220, 242, 200))
        file_a2.rechunk(chunks=(220, 242, 200))

        test = da.allclose(file_a1, a1)
        assert test.compute()

        test = da.allclose(file_a2, a2)
        assert test.compute()


def test_store_opti():
    """ Test the storing procedure with optimization.
    """

    def get_datasets(file_name, a1, a2):
        file_path = os.path.join(file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        f = h5py.File(file_path, 'w')
        dset1 = f.create_dataset('/data1', shape=a1.shape)
        dset2 = f.create_dataset('/data2', shape=a2.shape)
        return f, dset1, dset2

    output_dir = os.environ.get('OUTPUT_DIR')
    viz_file_name = "store" 

   
    # prepare test case
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
    new_config = CaseConfig(opti=None, 
                        scheduler_opti=None, 
                        out_path=None, 
                        buffer_size=5 * ONE_GIG, 
                        input_file_path=data, 
                        chunk_shape=None)
    new_config.create_or_overwrite(None, SUB_BIGBRAIN_SHAPE, overwrite=False)

    # ------ optimized
    arr = get_test_arr(new_config)
    a1 = arr[:220,:484,:400]
    a2 = arr[:220,:484,400:800]

    buffer_size = 5 * ONE_GIG
    dask.config.set({'optimizations': [optimize_func]})
    dask.config.set({'io-optimizer': {'memory_available': buffer_size,
                                        'scheduler_opti': True}})

    path = os.path.join(os.getenv('DATA_PATH'), "file1.hdf5")
    f, dset1, dset2 = get_datasets(path, a1, a2)
    s = da.store([a1, a2], [dset1, dset2], compute=False)
    s.compute()
    output_path = os.path.join(output_dir, viz_file_name + '_opti.png')
    # s.visualize(filename=output_path, optimize_graph=True)
    f.close()

    # verification
    dask.config.set({'optimizations': list()})
    dask.config.set({'io-optimizer': {'memory_available': buffer_size,
                                        'scheduler_opti': False}})

    a1 = arr[:220,:484,:400]
    a2 = arr[:220,:484,400:800]
    with h5py.File(path) as f:
        file_a1 = da.from_array(f['/data1'])
        file_a2 = da.from_array(f['/data2'])

        file_a1.rechunk(chunks=(220, 242, 200))
        file_a2.rechunk(chunks=(220, 242, 200))

        test = da.allclose(file_a1, a1)
        assert test.compute()

        test = da.allclose(file_a2, a2)
        assert test.compute()


if __name__ == "__main__":
    test_store_non_opti()