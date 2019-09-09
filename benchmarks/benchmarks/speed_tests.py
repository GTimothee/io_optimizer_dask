import sys
import os
import copy
import time
import numpy as np
import csv
import h5py
from random import shuffle

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


class Test_config():
    """ Contains the configuration for a test.
    """
    def __init__(self, opti, scheduler_opti, out_file_path, buffer_size):
        self.test_case = None
        self.opti = opti 
        self.scheduler_opti = scheduler_opti
        self.out_file_path = out_file_path
        self.buffer_size = buffer_size

        # default to not recreate file
        self.chunk_shape = None 
        self.shape = None
        self.overwrite = None

    def sum_case(self, nb_chunks):
        self.test_case = 'sum'
        self.nb_chunks = nb_chunks

    def create_or_overwrite(self, chunk_shape, shape, overwrite):
        self.chunk_shape = chunk_shape
        self.shape = shape
        self.overwrite = overwrite


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

    def run_test(config):
        """ Run a test given a specific configuration.
        """
        if config.test_case != 'sum':
            raise ValueError("Configuration not prepared for sum test case.")

        # get array
        file_path = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
        arr = get_test_arr(file_path, 
                           chunk_shape=config.chunk_shape, 
                           shape=config.shape, 
                           test_case='sum', 
                           nb_chunks=config.nb_chunks, 
                           overwrite=config.overwrite)

        # run test
        with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:
            res_dask, t = run(config.scheduler_opti, 
                              config.opti, 
                              arr, 
                              config.buffer_size)

        # write output
        out_file_path = os.path.join(out_path, 'non_opti.html')
        visualize([prof, rprof, cprof], config.out_file_path)
        writer.writerow([config.opti, 
                         config.scheduler_opti, 
                         config.chunk_shape, 
                         config.nb_chunks, 
                         config.buffer_size, t])

    # create the tests to be run
    # create the output directories
    output_dir = os.environ.get('OUTPUT_BENCHMARK_DIR')  
    tests = list()
    for is_scheduler in [True, False]:
        scheduler_status = 'scheduler_on' if is_scheduler else 'scheduler_off'
        sched_out_path = add_dir(output_dir, scheduler_status)

        for chunk_shape in shapes_to_test: 
            chunk_path = add_dir(sched_out_path, chunk_shape)
            
            for nb_chunks in chunks_to_test[chunk_shape][scheduler_status]:
                out_path = add_dir(chunk_path, str(nb_chunks) + '_chunks')

                if non_opti:
                    new_config = Test_config(False, False, out_file_path, buffer_size)
                    new_config.sum_case(nb_chunks)
                    tests.append(new_config)

                if opti:
                    new_config = Test_config(True, scheduler_opti, out_file_path, buffer_size)
                    new_config.sum_case(nb_chunks)
                    tests.append(new_config)

    # run the tests
    with open(os.path.join(output_dir, 'sum_speeds.csv'), mode='a+') as csv_out:
        writer = csv.writer(csv_out, delimiter=',')
        writer.writerow(['optimized', 'is_scheduler', 'chunk_shape', 
                         'nb_chunks_to_sum', 'buffer_size', 'processing_time'])
        
        shuffle(tests)
        for config in tests:
            run_test(config)


if __name__ == '__main__':
    chunks_to_test = {
        'slabs_dask_interpol': {
            'scheduler_on': [105, 210],
            'scheduler_off': [105]},

        'slabs_previous_exp': {
            'scheduler_on': [105],
            'scheduler_off': [105]},

        'blocks_dask_interpol':{
            'scheduler_on': [105, 210],
            'scheduler_off': [105]}, 

        'blocks_previous_exp': {
            'scheduler_on': [105],
            'scheduler_off': [105]}
    }

    non_opti, opti = (True, True)
    buffer_size = 5 * ONE_GIG
    shapes_to_test = ["blocks_dask_interpol", "slabs_dask_interpol"] 
    _sum(non_opti, opti, buffer_size, shapes_to_test, chunks_to_test)   