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

from tests_utils import *
from optimize_io.main import optimize_func

from test import Test
import pdb


def get_arr():
        """ get dask array from file
        """
        file_path = "/run/media/user/HDD 1TB/data/1.hdf5"
        chunks_shape = (1400, 1400, 700)
        arr = get_dask_array_from_hdf5(file_path, logic_chunks_shape=chunks_shape)
        arr = arr[:chunks_shape[0], :chunks_shape[1], :chunks_shape[2]]
        return arr


def manual_config_dask(buffer_size=ONE_GIG, opti=True, sched_opti=True):
    print(f'opti: {opti}\nsched_opti: {sched_opti}')
    opti_funcs = [optimize_func] if opti else list()
    print(f'opti_funcs: {opti_funcs} because opti:{opti}')
    
    dask.config.set({'optimizations': opti_funcs,
                     'io-optimizer': {
                        'memory_available': buffer_size,
                        'scheduler_opti': sched_opti}
    })


def write_npy_stack():
    flush_cache()
    buffer_size = 5.5 * ONE_GIG
    manual_config_dask(buffer_size=buffer_size, opti=True, sched_opti=True)

    # loading raw data if needed
    with dask.config.set(scheduler='single-threaded'):
        # load compressed data
        t = time.time()
        arr = get_arr()
        print(f'time to load compressed file in cache: {time.time() - t}')

        # write to numpy stack
        out_dir = 'data/out_3_numpy'
        t = time.time()
        a__, b__, c__ = da.to_npy_stack(out_dir, arr, axis=0)
        _ = dask.base.compute_as_if_collection(a__, b__, c__)
        print(f'time to save the array to numpy stack: {time.time() - t}')

        manual_config_dask(buffer_size=buffer_size, opti=False, sched_opti=False)

def write_hdf5():
    flush_cache()
    buffer_size = 5.5 * ONE_GIG

    # loading raw data if needed
    with dask.config.set(scheduler='single-threaded'):
        # load compressed data
        t = time.time()
        arr = get_arr()
        print(f'time to load compressed file in cache: {time.time() - t}')

        # write to numpy stack
        out_filepath = 'data/out.hdf5'
        if os.path.isfile(out_filepath):
            os.remove(out_filepath)
        manual_config_dask(buffer_size=buffer_size, opti=True, sched_opti=True)
        t = time.time()
        da.to_hdf5(out_filepath, 'data', arr, chunks=None, compression="gzip")
        print(f'time to save the array to hdf5 with compression: {time.time() - t}')

        manual_config_dask(buffer_size=buffer_size, opti=False, sched_opti=False)


def load_raw_write_hdf5():
    flush_cache()
    buffer_size = 5.5 * ONE_GIG

    # loading raw data if needed
    with dask.config.set(scheduler='single-threaded'):
        # load compressed data
        t = time.time()
        arr = get_arr()
        print(f'time to load compressed file in cache: {time.time() - t}')
        
        # load data in RAM      
        manual_config_dask(buffer_size=buffer_size, opti=False, sched_opti=False)
        t = time.time()
        raw_arr = arr.compute()
        print(f'time to load data in RAM: {time.time() - t}')

        # create dask array from data in RAM
        arr = da.from_array(raw_arr, chunks=(1400, 1400, 350))

        # write to numpy stack
        out_filepath = 'data/out.hdf5'
        if os.path.isfile(out_filepath):
            os.remove(out_filepath)
        manual_config_dask(buffer_size=buffer_size, opti=True, sched_opti=True)
        t = time.time()
        da.to_hdf5(out_filepath, 'data', arr, chunks=None)
        print(f'time to save the array to hdf5 without compression: {time.time() - t}')

        # write to numpy stack
        out_filepath = 'data/out.hdf5'
        os.remove(out_filepath)
        manual_config_dask(buffer_size=buffer_size, opti=True, sched_opti=True)
        t = time.time()
        da.to_hdf5(out_filepath, 'data', arr, chunks=None, compression="gzip")
        print(f'time to save the array to hdf5 with compression: {time.time() - t}')

        manual_config_dask(buffer_size=buffer_size, opti=False, sched_opti=False)


def load_raw_write_npy_stack():
    flush_cache()
    buffer_size = 5.5 * ONE_GIG
    manual_config_dask(buffer_size=buffer_size, opti=True, sched_opti=True)

    # loading raw data if needed
    with dask.config.set(scheduler='single-threaded'):
        # load compressed data
        t = time.time()
        arr = get_arr()
        print(f'time to load compressed file in cache: {time.time() - t}')
        
        # load data in RAM      
        t = time.time()
        manual_config_dask(buffer_size=buffer_size, opti=False, sched_opti=False)
        raw_arr = arr.compute()
        print(f'time to load data in RAM: {time.time() - t}')

        # create dask array from data in RAM
        arr = da.from_array(raw_arr, chunks=(1400, 1400, 350))

        # write to numpy stack
        out_dir = 'data/out_3_numpy'
        manual_config_dask(buffer_size=buffer_size, opti=True, sched_opti=True)
        t = time.time()
        a__, b__, c__ = da.to_npy_stack(out_dir, arr, axis=0)
        _ = dask.base.compute_as_if_collection(a__, b__, c__)
        print(f'time to save the array to numpy stack: {time.time() - t}')

        manual_config_dask(buffer_size=buffer_size, opti=False, sched_opti=False)


def load_raw_write_npy_file():
    flush_cache()

    # loading raw data if needed
    with dask.config.set(scheduler='single-threaded'):
        # load compressed data
        t = time.time()
        arr = get_arr()
        print(f'time to load compressed file in cache: {time.time() - t}')
        
        # load data in RAM
        t = time.time()
        raw_arr = arr.compute()
        print(f'time to load data in RAM: {time.time() - t}')
            
        # write to numpy file
        out_filepath = 'data/out_1.npy'
        if os.path.isfile(out_filepath):
            os.remove(out_filepath)
        t = time.time()
        np.save(out_filepath, raw_arr)
        print(f'time to save the array to numpy file: {time.time() - t}')

        manual_config_dask(buffer_size=buffer_size, opti=False, sched_opti=False)
            

def time_to_write_buffer_to_npy_files():
    """ create random array of 2.5GB 
    2.5GB = size of 1 buffer in experiment `small`
    measure time to write it to disk
    """
    def setup():
        x = da.random.random(size=(1400, 1400, 700))
        x = x.astype(np.float16)
        out_path = "/run/media/user/HDD 1TB/data/randomfile.hdf5"
        return x, out_path

    x, out_path = setup()
    raw_arr = x.compute()
    
    t = time.time()
    np.save(out_path, raw_arr)
    t = time.time() - t

    # timeit statement 
    print(f'time: {t}')

    manual_config_dask(buffer_size=buffer_size, opti=False, sched_opti=False)


if __name__ == "__main__":
    # time_to_write_buffer_to_npy_files()
    # load_raw_write_npy_file()
    # write_npy_stack()
    load_raw_write_hdf5()
    write_hdf5()