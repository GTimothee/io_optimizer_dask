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
    ----------
        dask_config: contains the test configuration
    """
    flush_cache()
    configure_dask(dask_config, optimize_func)
    # arr = get_test_arr(dask_config)
    a, b, c = get_test_arr(dask_config)
    try:
        t = time.time()
        with dask.config.set(scheduler='single-threaded'):
            _ = dask.base.compute_as_if_collection(a, b, c)
        # _ = arr.compute()
        t = time.time() - t
        return t
    except Exception as e:
        # print(traceback.format_exc())
        # print("An error occured during processing.")
        return 0
    

def run_test(writer, test, output_dir):
    """ Wrapper around 'run' function for diagnostics.

    Arguments:
    ----------
        writer:
        test:
        output_dir:
    """
    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:    
        t = run(getattr(test, 'dask_config'))
        uid = uuid.uuid4() 
        out_file_path = os.path.join(output_dir, 'output_imgs', str(uid) + '.html')
        out_file_path = None

        if t:
            visualize([prof, rprof, cprof], out_file_path)
            # pass

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
    """ Create all possible tests from a list of possible options.

    Arguments:
    ----------
        options: list of lists of configurations to try
    
    Returns 
    ----------
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

    """ Apply the split algorithm using Dask arrays.

    Arguments:
    ----------
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

    # csv_path = os.path.join(output_dir, 'exp1_' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '_out.csv')
    csv_path = os.path.join(output_dir, '_out.csv')
    csv_out, writer = create_csv_file(csv_path, columns, delimiter=',', mode='w+')

    if not debug_mode: 
        tests *= nb_repetitions
        shuffle(tests)
        
    nb_tests = len(tests)
    for i, test in enumerate(tests):
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

        print(f'\n\n[INFO] Processing test {i + 1}/{nb_tests} ~')
        test.print_config()
        run_test(writer, test, output_dir)
    csv_out.close()


def estimate_time():
    """ estimate the time to write one of the two buffers of 2.5GB each 
    in the experiment on 'small' array
    """
    import dask.array as da; 
    import numpy as np; 

    def setup():
        x = da.random.random(size=(1400, 1400, 700))
        x = x.astype(np.float16)
        out_path = "/run/media/user/HDD 1TB/data/randomfile.hdf5"
        return x, out_path

    x, out_path = setup()
    raw_arr = x.compute()
    
    t = time.time()
    da.to_hdf5(out_path, 'data', raw_arr, chunks=None)
    # np.save(out_path, raw_arr)
    t = time.time() - t

    # timeit statement 
    print(f'time: {t}')

if __name__ == "__main__":
    chunks_shapes = {
        "very_small":{
            "blocks":[(200, 200, 200)],
            "slabs":[(400, 400, 50)]
        },
        "small":{
            "blocks":[
                (700, 700, 700)],
            "slabs":[
                ("auto", 1400, 1400), #,]
                 (5, 1400, 1400),
                 (175, 1400, 1400)]
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
                (28, 3500, 3500),
                (50, 3500, 3500),]
                # (3500, 3500, 500)] -> 1 block ne rentre pas en mémoire
        }
    }

    """experiment(debug_mode=False,
        nb_repetitions=1,
        hardwares=["hdd"],
        cube_types=['small'],
        physical_chunked_options=[False],
        chunk_types=['slabs', 'blocks'],
        scheduler_options=[False],
        optimization_options=[True])"""
    estimate_time()



