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


def flush_cache():
    os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')


def register_profilers():
    rprof = ResourceProfiler()
    prof = Profiler()
    cacheprof = CacheProfiler()

    rprof.register()
    prof.register()
    cacheprof.register()


def unregister_profilers():
    rprof.unregister()
    prof.unregister()
    cacheprof.unregister()    


def run(is_scheduler, optimize, arr, buffer_size):
    """ Execute a dask array with or without optimization.
    
    Arguments:
        arr: dask_array
        optimize: should the optimization be activated
        buffer_size: size of the buffer for clustered strategy
        is_scheduler: activate scheduler optimization
    """
    flush_cache()

    # configuration
    if optimize:
        dask.config.set({'optimizations': optimize_func})
        dask.config.set({'io-optimizer': {
                            'memory_available': buffer_size,
                            'scheduler_opti': is_scheduler}
                            })
    else:
        dask.config.set({'optimizations': list()})

    # evaluation
    register_profilers()
    t = time.time()
    res = arr.compute()
    t = time.time() - t
    unregister_profilers()
    
    return res, t


def _sum(non_opti, opti, buffer_size, shapes_to_test, chunks_to_test):
    """ Test if the sum of n blocks yields the good result.

    Arguments:
        non_opti: test without optimization
        opti: test with optimization
        buffer_size: size of the buffer for optimization
        shapes_to_test: shapes that must be tested
        chunks_to_test: number of chunks to sum for each shapes
    """

    output_dir = os.environ.get('OUTPUT_BENCHMARK_DIR')
    data_path = get_test_array()
    with open(os.path.join(output_dir, 'sum_speeds.csv'), mode='a+') as csv_out:
        writer = csv.writer(csv_out, delimiter=',')
        writer.writerow(['optimized', 'is_scheduler', 'chunk_shape', 
                         'nb_chunks_to_sum', 'buffer_size', 'processing_time'])

        for is_scheduler in [False, True]:
            scheduler_status = 'scheduler_on' if is_scheduler else 'scheduler_off'
            sched_out_path = os.path.join(output_dir, scheduler_status)
            os.mkdir(sched_out_path, '0755') if not os.exists(sched_out_path)

            for chunk_shape in shapes_to_test: 
                chunk_path = os.path.join(sched_out_path, chunk_shape)
                os.mkdir(chunk_path, '0755') if not os.exists(sched_out_path)

                for nb_chunks in chunks_to_test[chunk_shape][scheduler_status]:
                    nb_chunk_path = os.path.join(chunk_path, str(nb_chunks) + '_chunks')
                    os.mkdir(nb_chunk_path, '0755') if not os.exists(sched_out_path)

                    if non_opti:
                        arr = get_test_arr(case='sum', nb_arr=nb_chunks)
                        res_dask, t = run(is_scheduler, False, nb_chunks, arr, buffer_size)
                        
                        out_file_path = os.path.join(nb_chunk_path, 'non_opti')
                        visualize([prof, rprof, cacheprof], out_file_path)

                        writer.writerow([False, is_scheduler, chunk_shape, nb_chunks, buffer_size, t])

                    if opti:
                        arr = get_test_arr(case='sum', nb_arr=nb_chunks)
                        res_opti, t = run(is_scheduler, True, nb_chunks, arr, buffer_size)

                        out_file_path = os.path.join(nb_chunk_path, 'non_opti')
                        visualize([prof, rprof, cacheprof], out_file_path)

                        writer.writerow([True, is_scheduler, chunk_shape, nb_chunks, buffer_size, t])

                    if opti and non_opti:
                        assert np.array_equal(res_dask, res_opti)


def _store(non_opti, opti):
    def get_datasets(file_name, a1, a2):
        file_path = os.path.join(file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        f = h5py.File(file_path, 'w')
        dset1 = f.create_dataset('/data1', shape=a1.shape)
        dset2 = f.create_dataset('/data2', shape=a2.shape)
        return f, dset1, dset2

    
    if non_opti:
        arr = get_test_arr()
        a1 = arr[:440,:,:]
        a2 = arr[:440,:,:]
        f, dset1, dset2 = get_datasets("file1.hdf5", a1, a2)
        arr = da.store([a1, a2], [dset1, dset2], compute=False)
        _, t = run(None, False, arr, buffer_size)
        f.close()
    

    if opti:
        for is_scheduler in [False, True]:
            arr = get_test_arr()
            a1 = arr[:440,:,:]
            a2 = arr[:440,:,:]
            f, dset1, dset2 = get_datasets("file1.hdf5", a1, a2)
            arr = da.store([a1, a2], [dset1, dset2], compute=False)
            _, t = run(True, is_scheduler, arr, buffer_size)
            f.close()


def benchmark():
    
    chunks_to_test = {
        'slabs_dask_interpol': {
            True: [105, 210],
            False: [105]},
        'slabs_previous_exp': {
            True: [105, 210],
            False: [105]},
        'blocks_dask_interpol':{
            True: [105, 210],
            False: [105]}, 
        'blocks_previous_exp': {
            True: [105, 210],
            False: [105]}
    }

    non_opti, opti = (True, False)
    buffer_size = 5 * ONE_GIG
    shapes_to_test = ["blocks_dask_interpol", "slabs_dask_interpol"]
    sum_scheduler_opti(non_opti, opti, buffer_size, shapes_to_test, chunks_to_test)   

benchmark()
# _store()