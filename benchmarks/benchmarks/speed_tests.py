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


cube_shapes = {
    "very_small": (400, 400, 400),
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
    },
    'very_small':{  # for local tests only
        'chunked': 4,
        'not_chunked': 5
    }
}

hardware_paths = {
    'ssd': os.getenv('SSD_PATH'),  # input and output dir
    'hdd': os.getenv('HDD_PATH')  # input and output dir
}


def run(config):
    """ Execute a dask array with a given configuration.
    
    Arguments:
        arr: dask_array
        config: contains the test configuration
    """

    flush_cache()
    configure_dask(config, optimize_func)
    arr = get_test_arr(config)

    t = time.time()
    res = arr.compute()
    t = time.time() - t
    return res, t


def run_test(writer, config, data):
    """ Wrapper around 'run' function
    """
    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:
        print(f'optimization enabled: {config.opti}')
        print(f'Processing cube ref: {data["cube_ref"]}')

        res_dask, t = run(config)

        # get diagnostics with dask diagnotics
        opti_info = 'opti' if config.opti else 'non_opti'
        out_file_path = os.path.join(data["output_path"], opti_info + '.html')
        visualize([prof, rprof, cprof], out_file_path)

        # write output in csv file
        writer.writerow([data["hardware"], 
                        data["cube_ref"],
                        data["chunk_type"],
                        data["chunks_shape"],
                        config.opti, 
                        config.scheduler_opti, 
                        config.buffer_size, 
                        t,
                        out_file_path])


def add_dir(path, new_dir):
    """ Create a new directory at path/new_dir
    """
    path = os.path.join(path, new_dir)
    if not os.path.exists(path):
        os.mkdir(path) 
    return path

"""
def _sum():"""
""" Test if the sum of n blocks yields the good result.

Arguments:
    non_opti: test without optimization
    opti: test with optimization
    buffer_size: size of the buffer for optimization
    shapes_to_test: shapes that must be tested
    chunks_to_test: number of chunks to sum for each shapes
"""

"""
'slabs_dask_interpol': {
        'scheduler_on': [105],
        'scheduler_off': [105]},

    'slabs_previous_exp': {
        'scheduler_on': [94],
        'scheduler_off': [94]},
"""
"""
    chunks_to_test = {
        'blocks_dask_interpol':{
            'scheduler_on': [210],
            'scheduler_off': [105]}, 

        'blocks_previous_exp': {
            'scheduler_on': [6],
            'scheduler_off': [3]}
    }

    non_opti, opti = (True, True)
    buffer_size = 5 * ONE_GIG
    shapes_to_test = ['blocks_previous_exp'] # list(chunks_to_test.keys())

    # create the tests to be run + create the output directories
    output_dir = os.environ.get('OUTPUT_BENCHMARK_DIR')  
    input_file_path = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
    tests = list()
    sum_dir = add_dir(output_dir, 'sum')
    for is_scheduler in [True, False]:
        scheduler_status = 'scheduler_on' if is_scheduler else 'scheduler_off'
        sched_out_path = add_dir(sum_dir, scheduler_status)

        for chunks_shape in shapes_to_test: 
            chunk_path = add_dir(sched_out_path, chunks_shape)
            
            for nb_chunks in chunks_to_test[chunks_shape][scheduler_status]:
                out_path = add_dir(chunk_path, str(nb_chunks) + '_chunks')

                if non_opti:
                    new_config = CaseConfig(False, False, out_path, buffer_size, input_file_path, first_exp_shapes[chunks_shape])
                    new_config.sum_case(nb_chunks)
                    tests.append(new_config)

                if opti:
                    new_config = CaseConfig(True, is_scheduler, out_path, buffer_size, input_file_path, first_exp_shapes[chunks_shape])
                    new_config.sum_case(nb_chunks)
                    tests.append(new_config)

    # run the tests
    with open(os.path.join(output_dir, 'sum_speeds.csv'), mode='w+') as csv_out:
        writer = csv.writer(csv_out, delimiter=',')
        writer.writerow(['optimized', 
                         'is_scheduler', 
                         'chunks_shape', 
                         'nb_chunks_to_sum', 
                         'buffer_size', 
                         'processing_time',
                         'output_file_path'])
        shuffle(tests)
        for config in tests:
            run_test(writer, config)"""


def load_json(file_path):
    """ Load a Python dictionary from a json file.
    """
    import json
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def create_tests_exp1(hardwares, cube_types, chunked_options, chunk_types, scheduler_options, optimization_options):
    """ Create tests to be run for experience 1
    """
    workspace = os.getenv('BENCHMARK_DIR')
    chunks_shapes = load_json(os.path.join(workspace, 'chunks_shapes.json'))

    buffer_sizes = {
        "very_small": ONE_GIG,
        "small": 5.5*ONE_GIG,
        "big": 15
    }

    tests = list()
    for hardware in hardwares: 

        # create dir
        work_dir = hardware_paths[hardware]
        exp_dir = add_dir(work_dir, 'experiment_1')

        for cube_type in cube_types: 

            for chunked in chunked_options:  # if we want the input array to be chunked or not
                
                # create dir
                chunk_status = "chunked" if chunked else "not_chunked"
                ref = cube_refs[cube_type][chunk_status]
                ref_dir = add_dir(exp_dir, str(ref))
                input_file_path = os.path.join(work_dir, str(ref) + '.hdf5')
                if not os.path.isfile(input_file_path):
                    create_input_arrays(input_file_path, cube_type)

                for chunk_type in chunk_types:
                    # create dir
                    chunk_type_path = add_dir(ref_dir, chunk_type)

                    for chunks_shape in chunks_shapes[cube_type][chunk_type]:
                        # create dir
                        out_path = add_dir(chunk_type_path, str(chunks_shape))
                        cube_chunks = chunks_shape if chunked == 'chunked' else None

                        for is_scheduler in scheduler_options:  
                            
                            # create dir
                            scheduler_status = 'scheduler_on' if is_scheduler else 'scheduler_off'
                            sched_path = add_dir(out_path, scheduler_status)

                            for opti in optimization_options:

                                for test_type in ["split"]:  # TODO: add a merge step
                                    # create dir
                                    algo_path = add_dir(sched_path, test_type)

                                    # create config
                                    new_config = CaseConfig(input_file_path, chunks_shape)
                                    
                                    new_config.optimization(opti, is_scheduler, buffer_sizes[cube_type])

                                    if test_type == "split":
                                        split_file_path = os.path.join(work_dir, "split.hdf5")
                                        new_config.split_case(input_file_path, split_file_path)
                                    else:
                                        raise ValueError("Not implemented yet.")

                                    meta_data = {
                                        "hardware": hardware, 
                                        "cube_ref": ref,
                                        "chunk_type": chunk_type,
                                        "chunks_shape": chunks_shape,
                                        "output_path": algo_path
                                    }
                                    tests.append((new_config, meta_data))
    return tests


def create_input_arrays(input_file_path, cube_type, chunks_shape=None, dtype="float16"):
    """
    arguments: 
        chunks_shape: physical chunk shape
    """
    dask_utils_perso.utils.create_random_cube(storage_type="hdf5",
                                                file_path=input_file_path,
                                                shape=cube_shapes[cube_type],  
                                                chunks_shape=chunks_shape,  
                                                dtype=dtype)


# TODO: add merge
def experiment_1():
    """ Applying the split and merge algorithms using Dask arrays.
    """
    tests = create_tests_exp1(hardwares=["ssd"], 
                            cube_types=['very_small'], 
                            chunked_options=[False], 
                            chunk_types=['blocks'], 
                            scheduler_options=[True], 
                            optimization_options=[True])
    
    output_dir = os.getenv('OUTPUT_BENCHMARK_DIR')
    with open(os.path.join(output_dir, 'experience_1_split.csv'), mode='w+') as csv_out:
        writer = csv.writer(csv_out, delimiter=',')
        writer.writerow(['hardware',
                         'ref',
                         'chunk_type',
                         'chunks_shape',
                         'opti',
                         'scheduler_opti',
                         'buffer_size', 
                         'processing_time',
                         'output_file_path'])

        shuffle(tests)
        for test in tests:
            config, data = test
            run_test(writer, config, data)


if __name__ == "__main__":
    experiment_1()