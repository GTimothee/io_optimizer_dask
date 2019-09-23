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
from tests_utils import get_test_arr, ONE_GIG


def sum():
    """ Test if the sum of two blocks yields the good
    result usign our optimization function.
    """
    output_dir = os.environ.get('OUTPUT_DIR')
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


"""def test_store():
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
    arr = get_test_arr()
    a1 = arr[:220,:484,:]
    a2 = arr[:220,484:968,:]
    
    f, dset1, dset2 = get_datasets("data/file1.hdf5", a1, a2)
    s = da.store([a1, a2], [dset1, dset2], compute=False)
    # s.compute()
    output_path = os.path.join(output_dir, file_name + '_non_opti.png')
    s.visualize(filename=output_path, optimize_graph=True)
    f.close()

    # ------ optimized
    arr = get_test_arr()
    a1 = arr[:220,:484,:]
    a2 = arr[:220,484:968,:]

    buffer_size = 5 * ONE_GIG
    dask.config.set({'optimizations': [optimize_func]})
    dask.config.set({'io-optimizer': {'memory_available': buffer_size,
                                        'scheduler_opti': True}})
    f, dset1, dset2 = get_datasets("data/file2.hdf5", a1, a2)
    s = da.store([a1, a2], [dset1, dset2], compute=False)
    # s.compute()
    output_path = os.path.join(output_dir, file_name + '_opti.png')
    s.visualize(filename=output_path, optimize_graph=True)
    f.close()"""