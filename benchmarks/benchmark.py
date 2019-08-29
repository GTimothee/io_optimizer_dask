import sys
import os
import copy
import time
import numpy as np
import csv

import dask

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


def _sum():
    """ Test if the sum of two blocks yields the good
    result usign our optimization function.
    """
    dask.config.set({'io-optimizer': {'memory_available': 5 * ONE_GIG,
                                        'scheduler_opti': False}})
    
    output_dir = os.environ.get('OUTPUT_BENCHMARK_DIR')
    data_path = get_test_array()
    key = 'data'
    with open(os.path.join(output_dir, 'speeds.csv'), mode='w+') as csv_out:
        writer = csv.writer(csv_out, delimiter=',')

        for nb_arr_to_sum in [245]:
            for chunk_shape in ['blocks_dask_interpol']:  

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
                writer.writerow(['non optimized', nb_arr_to_sum, t])

                """os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')
                dask.config.set({'optimizations': [optimize_func]})                
                result_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
                register_profilers()
                t2 = time.time()
                result_opti = result_opti.compute()
                t2 = time.time() - t2
                # visualize([prof, rprof, cacheprof])
                unregister_profilers()
                writer.writerow(['optimized', nb_arr_to_sum, t2])

                assert np.array_equal(result_non_opti, result_opti)

                # viz
                dask.config.set({'optimizations': []})
                result_non_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
                output_path = os.path.join(output_dir, 'test_non_opti' + chunk_shape + '.png')
                result_non_opti.visualize(filename=output_path, optimize_graph=True)

                dask.config.set({'optimizations': [optimize_func]})
                result_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
                output_path = os.path.join(output_dir, 'test_opti' + chunk_shape + '.png')
                result_opti.visualize(filename=output_path, optimize_graph=True)"""


def sum_scheduler_opti():
    """ Test if the sum of two blocks yields the good
    result usign our optimization function.
    """
    dask.config.set({'io-optimizer': {'memory_available': 3 * ONE_GIG,
                                        'scheduler_opti': True}})

    output_dir = os.environ.get('OUTPUT_BENCHMARK_DIR')
    data_path = get_test_array()
    key = 'data'
    with open(os.path.join(output_dir, 'speeds_opti_sched.csv'), mode='w+') as csv_out:
        writer = csv.writer(csv_out, delimiter=',')

        for nb_arr_to_sum in [70]:
            for chunk_shape in ['blocks_dask_interpol']:  # tests_utils.chunk_shapes:

                # test results
                """os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')
                dask.config.set({'optimizations': []})
                result_non_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
                register_profilers()
                t = time.time()
                result_non_opti = result_non_opti.compute()
                t = time.time() - t
                visualize([prof, rprof, cacheprof])# , os.path.join(output_dir, 'non_opti_&_schedule_profile_' + str(nb_arr_to_sum) + '.png'))
                unregister_profilers()
                writer.writerow(['non optimized', nb_arr_to_sum, t])"""

                os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')
                dask.config.set({'optimizations': [optimize_func]})                
                result_opti = get_test_arr(case='sum', nb_arr=nb_arr_to_sum)
                register_profilers()
                t2 = time.time()
                result_opti = result_opti.compute()
                t2 = time.time() - t2
                visualize([prof, rprof, cacheprof])# , os.path.join(output_dir, 'opti_&_schedule_profile_' + str(nb_arr_to_sum) + '.png'))
                unregister_profilers()
                writer.writerow(['optimized', nb_arr_to_sum, t2])
                
                # assert np.array_equal(result_non_opti, result_opti)


def benchmark():
    # _sum()
    sum_scheduler_opti()    


benchmark()