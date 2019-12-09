import sys
import os
import copy
import time
from time import gmtime, strftime
import numpy as np
import csv
import h5py
import itertools
import traceback
from random import shuffle
import uuid 

import dask.array as da
from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, visualize
from cachey import nbytes

from optimize_io.main import optimize_func
from tests_utils import *

from test import Test
import pdb


def run(dask_config):
    """ Execute a dask array with a given configuration.
    
    Arguments:
        config: contains the test configuration
        prod: whether to run the computations or not
    """
    flush_cache()
    configure_dask(dask_config, optimize_func)
    arr = get_test_arr(dask_config)
    try:
        t = time.time()
        _ = arr.compute()
        t = time.time() - t
        return t
    except Exception as e:
        print(traceback.format_exc())
        print("An error occured during processing.")
        return 0
    

def run_test(writer, test, output_dir):
    """ Wrapper around 'run' function for diagnostics.
    """
    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:    
        t = run(getattr(test, 'dask_config'))
        uid = uuid.uuid4() 
        # out_file_path = os.path.join(output_dir, str(uid) + '.html')
        out_file_path = None

        if t:
            # visualize([prof, rprof, cprof], out_file_path)
            pass

        writer.writerow([
            getattr(test, 'hardware'), 
            getattr(test, 'cube_ref'),
            getattr(test, 'chunk_type'),
            getattr(test, 'chunks_shape'),
            getattr(test, 'opti'), 
            getattr(test, 'scheduler_opti'), 
            getattr(test, 'buffer_size'), 
            t,
            out_file_path,
            uid 
        ])


def create_tests(options):
    """
    args:
        options: list of lists of configurations to try
    returns 
        A list of Test object containing the cartesian product of the combinations of "options"
    """
    def create_possible_tests(params):
        cube_type = params[1]
        chunk_type = params[3]
        test_list = list()
        for shape in chunks_shapes[cube_type][chunk_type]:
            if len(shape) != 3:
                print("Bad shape.")
                continue
            test_list.append(Test((*params, shape)))
        return test_list

    tests_params = [e for e in itertools.product(*options)]
    tests = list()
    for params in tests_params:
        if len(params) == 6:
            tests = tests + create_possible_tests(params)

    if not len(tests) > 0:
        print("Tests creation failed.")
        exit(1)
    return tests
        

def experiment(debug_mode,
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

    tests = create_tests([
        hardwares,
        cube_types,
        physical_chunked_options,
        chunk_types,
        scheduler_options,
        optimization_options,
    ])

    columns = ['hardware',
        'ref',
        'chunk_type',
        'chunks_shape',
        'opti',
        'scheduler_opti',
        'buffer_size', 
        'processing_time',
        'results_filepath'
    ]
    csv_path = os.path.join(output_dir, 'exp1_' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '_out.csv')
    csv_out, writer = create_csv_file(csv_path, columns, delimiter=',', mode='w+')

    if not debug_mode: 
        tests *= nb_repetitions
        shuffle(tests)
        
    for test in tests:
        # create array file if needed
        if not os.path.isfile(getattr(test, "array_filepath")):
            try:
                print(f'Creating input array...')
                create_random_cube(storage_type="hdf5",
                    file_path=getattr(test, 'array_filepath'),
                    shape=getattr(test, 'cube_shape'),  
                    physik_chunks_shape=getattr(test, 'physik_chunks_shape'), 
                    dtype=np.float16)
            except Exception as e:
                print(traceback.format_exc())
                print("Input array creation failed.")
                continue

        test.print_config()
        run_test(writer, test, output_dir)
    csv_out.close()


if __name__ == "__main__":
    chunks_shapes = {
        "very_small":{
            "blocks":[(200,200,200)],
            "slabs":[(50, 400, 400)]
        },
        "small":{
            "blocks":[
                (700, 700, 700)],
            "slabs":[
                (1400, 1400, "auto"),
                (1400, 1400, 5),
                (1400, 1400, 175)]
        },
        "big":{
            "blocks":[
                (350, 350, 350),
                (500, 500, 500),
                (875, 875, 875),],
                # (1750, 1750, 1750)], -> 1 block ne rentre pas en mémoire
            "slabs":[
                # (3500, 3500, "auto"), -> dont know size
                # (3500, 3500, 1), -> trop gros graph
                (3500, 3500, 28),
                (3500, 3500, 50),]
                # (3500, 3500, 500)] -> 1 block ne rentre pas en mémoire
        }
    }

    experiment(debug_mode=True,
        nb_repetitions=5,
        hardwares=["hdd"],
        cube_types=['small', 'big'],
        physical_chunked_options=[False],
        chunk_types=['blocks', 'slabs'],
        scheduler_options=[True, False],
        optimization_options=[True])