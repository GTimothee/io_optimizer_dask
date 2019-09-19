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
        config: contains the test configuration
    """
    flush_cache()

    if config.opti:
        opti_funcs = [optimize_func]
        scheduler_opti = config.scheduler_opti
        
    else:
        opti_funcs = list()
        scheduler_opti = False

    dask.config.set({'optimizations': opti_funcs})
    dask.config.set({'io-optimizer': {
                        'memory_available': config.buffer_size,
                        'scheduler_opti': scheduler_opti}
                        })

    print("at runtime: ", arr.shape, arr.chunks)

    # evaluation
    t = time.time()
    res = arr.compute()
    t = time.time() - t
    return res, t


class Test_config():
    """ Contains the configuration for a test.
    """
    def __init__(self, opti, scheduler_opti, out_path, buffer_size, input_file_path, chunk_shape):
        self.test_case = None
        self.opti = opti 
        self.scheduler_opti = scheduler_opti
        self.out_path = out_path
        self.buffer_size = buffer_size
        self.input_file_path = input_file_path
        self.chunk_shape = chunk_shape
        self.split_file = None

        # default to not recreate file
        self.shape = None
        self.overwrite = None

    def sum_case(self, nb_chunks):
        self.test_case = 'sum'
        self.nb_chunks = nb_chunks

    def create_or_overwrite(self, chunk_shape, shape, overwrite):
        self.chunk_shape = chunk_shape
        self.shape = shape
        self.overwrite = overwrite

    def split_case(self, hardware, ref, chunk_type, chunk_shape, split_file):
        self.cube_ref = ref
        self.test_case = 'split'
        self.hardware = hardware
        self.split_file = split_file
        if not self.chunk_shape:
            self.chunk_shape = chunk_shape
        self.chunk_type = chunk_type

    def write_output(self, writer, out_file_path, t):
        if self.test_case == 'sum':
            data = [
                self.opti, 
                self.scheduler_opti, 
                self.chunk_shape, 
                self.nb_chunks, 
                self.buffer_size, 
                t,
                out_file_path
            ]
        elif self.test_case == 'split':
            data = [
                self.hardware, 
                self.cube_ref,
                self.chunk_type,
                self.chunk_shape,
                self.opti, 
                self.scheduler_opti, 
                self.buffer_size, 
                t,
                out_file_path
            ]
        else:
            raise ValueError("Unsupported test case.")
        writer.writerow(data)


def run_test(writer, config):
        """ Get a test array, run the test and write the output.
        """
        print("chunk shape in config", config.chunk_shape)

        arr = get_test_arr(config.input_file_path, 
                           chunk_shape=config.chunk_shape, 
                           shape=config.shape, 
                           test_case=config.test_case, 
                           nb_chunks=config.nb_chunks, 
                           overwrite=config.overwrite,
                           split_file=config.split_file)

        # run test
        with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:
            res_dask, t = run(arr, config)

        # write output
        opti_info = 'opti' if config.opti else 'non_opti'
        out_file_path = os.path.join(config.out_path, opti_info + '.html')
        visualize([prof, rprof, cprof], out_file_path)
        config.write_output(writer, out_file_path, t)


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
            'scheduler_on': [105],
            'scheduler_off': [105]},

        'slabs_previous_exp': {
            'scheduler_on': [94],
            'scheduler_off': [94]},

        'blocks_dask_interpol':{
            'scheduler_on': [105],
            'scheduler_off': [105]}, 

        'blocks_previous_exp': {
            'scheduler_on': [105],
            'scheduler_off': [105]}
    }

    non_opti, opti = (False, True)
    buffer_size = 5 * ONE_GIG
    shapes_to_test = ["slabs_previous_exp"] 

    # create the tests to be run
    # create the output directories
    output_dir = os.environ.get('OUTPUT_BENCHMARK_DIR')  
    input_file_path = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
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
                    new_config = Test_config(False, False, out_path, buffer_size, input_file_path, first_exp_shapes[chunk_shape])
                    new_config.sum_case(nb_chunks)
                    tests.append(new_config)

                if opti:
                    new_config = Test_config(True, is_scheduler, out_path, buffer_size, input_file_path, first_exp_shapes[chunk_shape])
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
            run_test(writer, config)


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

    hardware_paths = {
        'ssd': os.getenv('SSD_PATH'),  # input and output dir
        'hdd': os.getenv('HDD_PATH')  # input and output dir
    }
    
    workspace = os.getenv('BENCHMARK_DIR')
    chunk_shapes = load_json(os.path.join(workspace, 'chunk_shapes.json'))

    non_opti, opti = (True, True)
    buffer_size = 5 * ONE_GIG

    # create tests
    tests = list()
    for hardware in ["hdd"]: # [ssd_path, hdd_path]:
        data_path = hardware_paths[hardware]
        split_dir = add_dir(data_path, 'split')

        for cube_type in ['small']: # ['small', 'big']:
            cube_dir = add_dir(split_dir, cube_type)

            for chunked in [False]: # [False, True]:
                chunk_status = 'chunked' if chunked else 'not_chunked'
                auto_chunk = True if chunked else None 
                ref_dir = add_dir(cube_dir, chunk_status)
                ref = cube_refs[cube_type][chunk_status]
                input_file_path = os.path.join(data_path, str(ref) + '.hdf5')

                for chunk_type in ['blocks', 'slabs']:
                    chunk_type_path = add_dir(ref_dir, chunk_type)

                    for chunk_shape in chunk_shapes[cube_type][chunk_type]:
                        out_path = add_dir(chunk_type_path, str(chunk_shape))

                        for is_scheduler in [True, False]:
                            scheduler_status = 'scheduler_on' if is_scheduler else 'scheduler_off'
                            sched_path = add_dir(out_path, scheduler_status)

                            split_file_path = os.path.join(data_path, 'split_file.hdf5')
                            if os.path.isfile(split_file_path):
                                os.remove(split_file_path)
                            split_file = h5py.File(split_file_path, 'w') 

                            if non_opti:
                                new_config = Test_config(False, False, sched_path, buffer_size, input_file_path)
                                config.create_or_overwrite(auto_chunk, cube_shapes[cube_type], overwrite=False)
                                new_config.split_case(hardware, ref, chunk_type, chunk_shape, split_file)
                                tests.append(new_config)

                            if opti:
                                new_config = Test_config(True, is_scheduler, sched_path, buffer_size, input_file_path)
                                config.create_or_overwrite(auto_chunk, cube_shapes[cube_type], overwrite=False)
                                new_config.split_case(hardware, ref, chunk_type, chunk_shape, split_file)
                                tests.append(new_config)

    # run the tests
    output_dir = os.getenv('OUTPUT_BENCHMARK_DIR')
    with open(os.path.join(output_dir, 'split_speeds.csv'), mode='w+') as csv_out:
        writer = csv.writer(csv_out, delimiter=',')
        writer.writerow(['hardware',
                         'ref',
                         'chunk_type',
                         'chunk_shape',
                         'opti',
                         'scheduler_opti',
                         'buffer_size', 
                         'processing_time',
                         'output_file_path'])
        shuffle(tests)
        for config in tests:
            run_test(config)

_sum()
# experiment_1()