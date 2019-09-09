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


# shapes used for the first experiment (assessing need for dask array optimization)
first_exp_shapes = {'slabs_dask_interpol': ('auto', (1210), (1400)), 
                    'slabs_previous_exp': (7, (1210), (1400)),
                    'blocks_dask_interpol': (220, 242, 200), 
                    'blocks_previous_exp': (770, 605, 700)}


def run(is_scheduler, optimize, arr, buffer_size):
    """ Execute a dask array with or without optimization.
    
    Arguments:
        arr: dask_array
        optimize: should the optimization be activated
        buffer_size: size of the buffer for clustered strategy
        is_scheduler: activate scheduler optimization
    """
    flush_cache()

    if optimize:
        dask.config.set({'optimizations': [optimize_func]})
        dask.config.set({'io-optimizer': {
                            'memory_available': buffer_size,
                            'scheduler_opti': is_scheduler}
                            })
    else:
        dask.config.set({'optimizations': []})
        dask.config.set({'io-optimizer': {
                            'memory_available': buffer_size,
                            'scheduler_opti': False}
                            })

    # evaluation
    t = time.time()
    res = arr.compute()
    t = time.time() - t
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

    def add_dir(workspace, new_dir):
        path = os.path.join(workspace, new_dir)
        if not os.path.exists(path):
            os.mkdir(path) 
        return path

    output_dir = os.environ.get('OUTPUT_BENCHMARK_DIR')
    data_path = get_test_array()
    with open(os.path.join(output_dir, 'sum_speeds.csv'), mode='a+') as csv_out:
        writer = csv.writer(csv_out, delimiter=',')
        writer.writerow(['optimized', 'is_scheduler', 'chunk_shape', 
                         'nb_chunks_to_sum', 'buffer_size', 'processing_time'])

        for is_scheduler in [True]:
            scheduler_status = 'scheduler_on' if is_scheduler else 'scheduler_off'
            sched_out_path = add_dir(output_dir, scheduler_status)

            for chunk_shape in shapes_to_test: 
                chunk_path = add_dir(sched_out_path, chunk_shape)
                
                for nb_chunks in chunks_to_test[chunk_shape][scheduler_status]:
                    out_path = add_dir(chunk_path, str(nb_chunks) + '_chunks')

                    if non_opti:
                        arr = get_test_arr(case='sum', nb_arr=nb_chunks)
                        out_file_path = os.path.join(out_path, 'non_opti.html')
                        with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:
                            res_dask, t = run(is_scheduler, False, arr, buffer_size)
                        visualize([prof, rprof, cprof], out_file_path)
                        writer.writerow([False, is_scheduler, chunk_shape, nb_chunks, buffer_size, t])

                    if opti:
                        arr = get_test_arr(case='sum', nb_arr=nb_chunks)
                        out_file_path = os.path.join(out_path, 'opti.html')
                        with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:
                            res_opti, t = run(is_scheduler, True, arr, buffer_size)
                        visualize([prof, rprof, cprof], out_file_path)
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
            _, t = run(is_scheduler, True, arr, buffer_size)
            f.close()


def benchmark():
    chunks_to_test = {
        'slabs_dask_interpol': {
            'scheduler_on': [105, 210],
            'scheduler_off': [105]},

        'slabs_previous_exp': {
            'scheduler_on': [105, 210],
            'scheduler_off': [105]},

        'blocks_dask_interpol':{
            'scheduler_on': [105, 210],
            'scheduler_off': [105]}, 

        'blocks_previous_exp': {
            'scheduler_on': [105, 210],
            'scheduler_off': [105]}
    }

    non_opti, opti = (True, True)
    buffer_size = 5 * ONE_GIG
    shapes_to_test = ["blocks_dask_interpol", "slabs_dask_interpol"] #, "slabs_dask_interpol"]
    _sum(non_opti, opti, buffer_size, shapes_to_test, chunks_to_test)   

if __name__ == '__main__':
    benchmark()
# _store()