import sys
import os
import copy
import time
import numpy as np
import csv
import h5py
import itertools
import traceback
from random import shuffle

import dask
import dask.array as da
from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, visualize
from cachey import nbytes

import optimize_io
from optimize_io.main import optimize_func
import tests_utils
from tests_utils import *

from test import Test
from utils import create_csv_file


def run(config):
    """ Execute a dask array with a given configuration.
    
    Arguments:
        config: contains the test configuration
        prod: whether to run the computations or not
    """
    flush_cache()
    configure_dask(config, optimize_func)
    arr = get_test_arr(config)

    try:
        t = time.time()
        _ = arr.compute()
        t = time.time() - t
        return t
    except Exception as e:
        print(traceback.format_exc())
    return None
    


def run_test(writer, test):
    """ Wrapper around 'run' function
    """
    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:
        print(f'optimization enabled: {test.opti}')
        print(f'Processing cube ref: {test.cube_ref}')
    
        if run(config):
            # create diagnostics file
            opti_info = 'opti' if test.opti else 'non_opti'
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


def create_tests_exp1(options):
    """
    args:
        options: list of lists of configurations to try
    returns 
        A list of Test object containing the cartesian product of the combinations of "options"
    """
    tests_params = [e for e in itertools.product(*options)]
    tests = [Test(params) for params in tests_params if len(params) == 6]
    if not len(tests) > 0:
        print("Tests creation failed.")
        exit(1)
    return tests
        

def experiment_1(debug_mode,
    nb_repetitions,
    hardwares,
    cube_types,
    physical_chunked_options,
    chunk_types,
    scheduler_options,
    optimization_options):

    """ Applying the split algorithm using Dask arrays.

    args:
        nb_repetitions,
        hardwares,
        cube_types,
        physical_chunked_options,
        chunk_types,
        scheduler_options,
        optimization_options
        debug_mode: 
            False means need to run the tests (with repetitions etc.). 
            True means we will try the algorithm to see if the graph has been optimized as we wanted. 
            Dont run the actual tests, just the optimization.
    """

    output_dir = os.getenv('OUTPUT_BENCHMARK_DIR')
    workspace = os.getenv('BENCHMARK_DIR')
    cs_filepath = os.path.join(workspace, 'chunks_shapes.json')
    out_filepath = os.path.join(output_dir, 'exp1_out.csv')

    tests = create_tests_exp1([
        hardwares,
        cube_types,
        physical_chunked_options,
        chunk_types,
        scheduler_options,
        optimization_options
    ])

    columns = ['hardware',
        'ref',
        'chunk_type',
        'chunks_shape',
        'opti',
        'scheduler_opti',
        'buffer_size', 
        'processing_time',
        'results_filepath',
        'log_time'
    ]
    csv_path = os.path.join(output_dir, 'exp1_out.csv')
    csv_out, writer = create_csv_file(csv_path, columns, delimiter=',', mode='w+')

    if not debug_mode: 
        tests *= nb_repetitions

    shuffle(tests)
    for test in tests:
        # create array file if needed
        if notn debug_mode and not os.path.isfile(array_filepath):
            try:
                dask_utils_perso.utils.create_random_cube(storage_type="hdf5",
                    file_path=test.array_file_path,
                    shape=test.cube_shape,  
                    chunks_shape=test.physik_chunked, 
                    dtype=np.float16)
            except:
                print("Input array creation failed.")
                continue

        # run_test(writer, test)

    csv_out.close()


if __name__ == "__main__":
    experiment_1(debug_mode=True,
        nb_repetitions=5,
        hardwares=["ssd"],
        cube_types=['big', 'small'],
        physical_chunked_options=[False],
        chunk_types=['blocks', 'slabs'],
        scheduler_options=[True, False],
        optimization_options=[True, False])