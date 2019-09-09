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


__all__ = ['_sum',
           'experiment_1']


# shapes used for the first experiment (assessing need for dask array optimization)
first_exp_shapes = {'slabs_dask_interpol': ('auto', (1210), (1400)), 
                    'slabs_previous_exp': (7, (1210), (1400)),
                    'blocks_dask_interpol': (220, 242, 200), 
                    'blocks_previous_exp': (770, 605, 700)}


def run(arr, config):
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
                            'memory_available': config.buffer_size,
                            'scheduler_opti': config.scheduler_opti}
                            })
    else:
        dask.config.set({'optimizations': []})
        dask.config.set({'io-optimizer': {
                            'memory_available': config.buffer_size,
                            'scheduler_opti': False}
                            })

    # evaluation
    t = time.time()
    if config.test_case == 'sum':
        res = arr.compute()
    elif config.test_case == 'split':
        # res.store(config.in_arrays)
    t = time.time() - t
    return res, t


class Test_config():
    """ Contains the configuration for a test.
    """
    def __init__(self, opti, scheduler_opti, out_path, buffer_size):
        self.test_case = None
        self.opti = opti 
        self.scheduler_opti = scheduler_opti
        self.out_path = out_path
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

    def split_case(self):
        self.test_case = 'split'


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
            res_dask, t = run(arr, config)

        # write output
        out_file_path = os.path.join(config.out_path, 'non_opti.html')
        visualize([prof, rprof, cprof], out_file_path)
        writer.writerow([config.opti, 
                         config.scheduler_opti, 
                         config.chunk_shape, 
                         config.nb_chunks, 
                         config.buffer_size, 
                         t,
                         out_file_path])


def add_dir(workspace, new_dir):
        path = os.path.join(workspace, new_dir)
        if not os.path.exists(path):
            os.mkdir(path) 
        return path


def _sum():
    """ Test if the sum of n blocks yields the good result.

    Arguments:
        non_opti: test without optimization
        opti: test with optimization
        buffer_size: size of the buffer for optimization
        shapes_to_test: shapes that must be tested
        chunks_to_test: number of chunks to sum for each shapes
    """

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

    # create the tests to be run
    # create the output directories
    output_dir = os.environ.get('OUTPUT_BENCHMARK_DIR')  
    tests = list()
    sum_dir = add_dir(output_dir, 'sum')
    for is_scheduler in [True, False]:
        scheduler_status = 'scheduler_on' if is_scheduler else 'scheduler_off'
        sched_out_path = add_dir(sum_dir, scheduler_status)

        for chunk_shape in shapes_to_test: 
            chunk_path = add_dir(sched_out_path, chunk_shape)
            
            for nb_chunks in chunks_to_test[chunk_shape][scheduler_status]:
                out_path = add_dir(chunk_path, str(nb_chunks) + '_chunks')

                if non_opti:
                    new_config = Test_config(False, False, out_path, buffer_size)
                    new_config.sum_case(nb_chunks)
                    tests.append(new_config)

                if opti:
                    new_config = Test_config(True, is_scheduler, out_path, buffer_size)
                    new_config.sum_case(nb_chunks)
                    tests.append(new_config)

    # run the tests
    with open(os.path.join(output_dir, 'sum_speeds.csv'), mode='w+') as csv_out:
        writer = csv.writer(csv_out, delimiter=',')
        writer.writerow(['optimized', 
                         'is_scheduler', 
                         'chunk_shape', 
                         'nb_chunks_to_sum', 
                         'buffer_size', 
                         'processing_time',
                         'output_file_path'])
        shuffle(tests)
        for config in tests:
            run_test(config)


def load_json(file_path):
    """ Load a Python dictionary from a json file.
    """
    import json
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


# TODO: add split part
# TODO: add merge part
def experiment_1():
    """ Applying the split and merge algorithms using Dask arrays.
    """

    def create_cube(data_path, ref, exp_id, chunked):
        if not os.path.isfile(file_path):
            auto_chunk = True if chunked else None 
            create_random_cube(storage_type="hdf5",
                        file_path=file_path,
                        shape=cube_shapes[exp_id],
                        chunks_shape=auto_chunk,
                        dtype="float16")
        return file_path


    def split():
        return


    def merge():
        return


    cube_shapes = {
        "small": (1400, 1400, 1400), 
        "big": (3500, 3500, 3500),
    }

    cube_refs = {
        'small': {
            'chunked': 0,  # chunked mean physical chunk here
            'not_chunked': 1 
        },
        'big': {
            'chunked': 2,  # chunked mean physical chunk here
            'not_chunked': 3 
        }
    }

    ssd_path = os.getenv('SSD_PATH')  # input and output dir
    hdd_path = os.getenv('HDD_PATH')  # input and output dir
    output_dir = os.getenv('OUTPUT_BENCHMARK_DIR')
    chunk_shapes = load_json(os.path.join(workspace, 'chunk_shapes.json'))  # for the output csv file only

    # create tests
    tests = list()
    for output_dir in [ssd_path, hdd_path]:
        split_dir = add_dir(output_dir, 'split')

        for cube_id in ['small', 'big']:

            for chunked in [False, True]:
                # create cube if does not exist
                ref = cube_refs[cube_id][chunked]
                file_path = create_cube(data_path, ref, cube_id, chunked)

                for chunk_type in ['blocks', 'slabs']:
                    chunk_type_path = add_dir(sched_out_path, chunk_type)

                    for shape in chunk_shapes[cube_id][chunk_type]:
                        out_path = add_dir(chunk_type_path, str(shape))

                        for is_scheduler in [True, False]:
                            scheduler_status = 'scheduler_on' if is_scheduler else 'scheduler_off'
                            sched_out_path = add_dir(split_dir, scheduler_status)
                        
                            if non_opti:
                                new_config = Test_config(False, False, out_path, buffer_size)
                                new_config.split_case()
                                tests.append(new_config)

                            if opti:
                                new_config = Test_config(True, is_scheduler, out_path, buffer_size)
                                new_config.split_case()
                                tests.append(new_config)

    # run the tests
    with open(os.path.join(output_dir, 'split_speeds.csv'), mode='w+') as csv_out:
        writer = csv.writer(csv_out, delimiter=',')
        writer.writerow(['optimized', 
                         'is_scheduler', 
                         'chunk_shape', 
                         'buffer_size', 
                         'processing_time',
                         'output_file_path'])
        shuffle(tests)
        for config in tests:
            run_test(config)