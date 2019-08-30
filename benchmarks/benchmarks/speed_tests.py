import sys
import os
import copy
import time
import numpy as np
import csv
import h5py

import dask
import dask.array as da

import optimize_io
from optimize_io.main import optimize_func

import tests_utils
from tests_utils import *

from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, visualize
from cachey import nbytes


rprof = ResourceProfiler()
prof = Profiler()
cacheprof = CacheProfiler()


def register_profilers():
    rprof.register()
    prof.register()
    cacheprof.register()


def unregister_profilers():
    rprof.unregister()
    prof.unregister()
    cacheprof.unregister()    


def _sum(non_opti, opti, buffer_size):
    """ Test if the sum of two blocks yields the good
    result usign our optimization function.
    """
    dask.config.set({'io-optimizer': {'memory_available': buffer_size,
                                        'scheduler_opti': False}})
    
    output_dir = os.environ.get('OUTPUT_BENCHMARK_DIR')
    data_path = get_test_array()
    key = 'data'
    with open(os.path.join(output_dir, 'speeds.csv'), mode='a+') as csv_out:
        writer = csv.writer(csv_out, delimiter=',')

        for nb_arr_to_sum in [35]:
            for chunk_shape in tests_utils.chunk_shapes:  

                if non_opti:
                    # test results
                    os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')
                    dask.config.set({'optimizations': []})
                    result_non_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
                    register_profilers()
                    t = time.time()
                    result_non_opti = result_non_opti.compute()
                    t = time.time() - t
                    print("processing time", t, "seconds")
                    # visualize([prof, rprof, cacheprof])
                    unregister_profilers()
                    writer.writerow(['non optimized', chunk_shape, nb_arr_to_sum, t])

                if opti:
                    os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')
                    dask.config.set({'optimizations': [optimize_func]})                
                    result_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
                    register_profilers()
                    t2 = time.time()
                    result_opti = result_opti.compute()
                    t2 = time.time() - t2
                    # visualize([prof, rprof, cacheprof])
                    unregister_profilers()
                    writer.writerow(['optimized', chunk_shape, nb_arr_to_sum, t2])

                if opti and non_opti:
                    assert np.array_equal(result_non_opti, result_opti)


def sum_scheduler_opti(non_opti, opti, buffer_size):
    """ Test if the sum of two blocks yields the good
    result usign our optimization function.
    """
    dask.config.set({'io-optimizer': {'memory_available': buffer_size,
                                        'scheduler_opti': True}})

    output_dir = os.environ.get('OUTPUT_BENCHMARK_DIR')
    data_path = get_test_array()
    key = 'data'
    with open(os.path.join(output_dir, 'speeds_opti_sched.csv'), mode='a+') as csv_out:
        writer = csv.writer(csv_out, delimiter=',')

        for nb_arr_to_sum in [35]:
            for chunk_shape in tests_utils.chunk_shapes: 

                if non_opti:

                    # test results
                    os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')
                    dask.config.set({'optimizations': []})
                    result_non_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
                    register_profilers()
                    t = time.time()
                    result_non_opti = result_non_opti.compute()
                    t = time.time() - t
                    # visualize([prof, rprof, cacheprof])# , os.path.join(output_dir, 'non_opti_&_schedule_profile_' + str(nb_arr_to_sum) + '.png'))
                    unregister_profilers()
                    writer.writerow(['non optimized', chunk_shape, nb_arr_to_sum, t])
                
                if opti:
                    os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')
                    dask.config.set({'optimizations': [optimize_func]})                
                    result_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
                    register_profilers()
                    t2 = time.time()
                    result_opti = result_opti.compute()
                    t2 = time.time() - t2
                    # visualize([prof, rprof, cacheprof])# , os.path.join(output_dir, 'opti_&_schedule_profile_' + str(nb_arr_to_sum) + '.png'))
                    unregister_profilers()
                    writer.writerow(['optimized', chunk_shape, nb_arr_to_sum, t2])
                    
                if opti and non_opti:
                    assert np.array_equal(result_non_opti, result_opti)


def _store():
    def get_datasets(file_name, a1, a2):
        file_path = os.path.join(file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        f = h5py.File(file_path, 'w')
        dset1 = f.create_dataset('/data1', shape=a1.shape)
        dset2 = f.create_dataset('/data2', shape=a2.shape)
        return f, dset1, dset2

    # non optimized
    os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')
    print("non optimized")
    arr = get_test_arr()
    a1 = arr[:440,:,:]
    a2 = arr[:440,:,:]

    f, dset1, dset2 = get_datasets("file1.hdf5", a1, a2)
    s = da.store([a1, a2], [dset1, dset2], compute=False)
    register_profilers()
    t = time.time()
    s.compute()
    t = time.time() - t
    visualize([prof, rprof, cacheprof])
    unregister_profilers()
    print("processing time", t)
    f.close()

    # optimized
    os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')
    print("optimized")
    buffer_size = 5 * ONE_GIG
    dask.config.set({'optimizations': [optimize_func]})
    dask.config.set({'io-optimizer': {'memory_available': buffer_size,
                                        'scheduler_opti': True}})

    arr = get_test_arr()
    a1 = arr[:440,:,:]
    a2 = arr[:440,:,:]

    f, dset1, dset2 = get_datasets("file2.hdf5", a1, a2)
    s = da.store([a1, a2], [dset1, dset2], compute=False)
    register_profilers()
    t = time.time()
    s.compute()
    t = time.time() - t
    visualize([prof, rprof, cacheprof])
    unregister_profilers()
    print("processing time", t)
    f.close()


def benchmark():
    buffer_size = 5 * ONE_GIG
    non_opti, opti = (True, True)
    _sum(non_opti, opti, buffer_size)

    buffer_size = 5 * ONE_GIG
    non_opti, opti = (True, True)
    sum_scheduler_opti(non_opti, opti, buffer_size)    


_store()