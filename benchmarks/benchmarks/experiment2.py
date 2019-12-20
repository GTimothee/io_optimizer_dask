import os
import time
import numpy as np
import pdb

import dask.array as da
from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, visualize
from cachey import nbytes

from tests_utils import *
from optimize_io.main import optimize_fun


a1 = {
    'name': '1_normal_compressed',
    'physik_cs': None,
    'distrib': 'normal',
    'compression': 'gzip'
}

a2 = {
    'name': '1_normal_uncompressed',
    'physik_cs': None,
    'distrib': 'normal',
    'compression': None
}

a3 = {
    'name': '1_uniform_compressed',
    'physik_cs': None,
    'distrib': 'uniform',
    'compression': 'gzip'
}

a4 = {
    'name': '1_uniform_uncompressed',
    'physik_cs': None,
    'distrib': 'uniform',
    'compression': None
}


test_arrays = [a1, a2, a3, a4]


def onthefly_to_nps():
    print('Writing to npy stack file without loading raw data in RAM.')

    out_dir = 'data/out_3_numpy'
    out_file_path = "outputs/write_npy_stack.html"

    # write to numpy stack
    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:    
        t = time.time()

        write_to_npy_stack(out_dir, arr)

        print(f'time to save the array to numpy stack: {time.time() - t}')
        visualize([prof, rprof, cprof], out_file_path)


def onthefly_to_hdf5():
    print('Writing to hdf5 file without loading raw data in RAM.')

    # write to numpy stack
    out_filepath = 'data/out.hdf5'
    if os.path.isfile(out_filepath):
        os.remove(out_filepath)

    out_file_path = "outputs/write_hdf5.html"
    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof: 
        t = time.time()

        da.to_hdf5(out_filepath, 'data', arr, chunks=(1400,1400,350), compression="gzip")

        print(f'time to save the array to hdf5 with compression: {time.time() - t}')
        visualize([prof, rprof, cprof], out_file_path)


def uncompress_to_hdf5():
    print('Writing to hdf5 file after loading raw data in RAM.')
    
    raw_arr = uncompress()

    # create dask array from data in RAM
    arr = da.from_array(raw_arr, chunks=(1400, 1400, 350))

    # write to numpy stack
    out_filepath = 'data/out.hdf5'
    if os.path.isfile(out_filepath):
        os.remove(out_filepath)

    out_file_path = "outputs/load_raw_write_hdf5_uncompressed.html"
    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof: 
        t = time.time()

        da.to_hdf5(out_filepath, 'data', arr, chunks=None)

        print(f'time to save the array to hdf5 without compression: {time.time() - t}')
        visualize([prof, rprof, cprof], out_file_path)

    # write to numpy stack
    out_filepath = 'data/out.hdf5'
    os.remove(out_filepath)

    out_file_path = "outputs/load_raw_write_hdf5_commpressed.html"
    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof: 
        t = time.time()

        da.to_hdf5(out_filepath, 'data', arr, chunks=None, compression="gzip")

        print(f'time to save the array to hdf5 with compression: {time.time() - t}')
        visualize([prof, rprof, cprof], out_file_path)



def uncompress_to_nps():
    print('Writing to numpy stack after loading raw data in RAM.')
    
    # load data in RAM      
    raw_arr = uncompress()

    # create dask array from data in RAM
    arr = da.from_array(raw_arr, chunks=(1400, 1400, 350))

    # write to numpy stack
    out_dir = 'data/out_numpy'

    out_file_path = "outputs/load_raw_write_npy_stack.html"
    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof: 
        t = time.time()

        write_to_npy_stack(out_dir, arr)

        print(f'time to save the array to numpy stack: {time.time() - t}')
        visualize([prof, rprof, cprof], out_file_path)


def uncompress_to_npy():
    print('Writing to numpy file after loading raw data in RAM.')
    out_filepath = 'data/out_1.npy'
    diagnostics_filepath = "outputs/load_raw_write_npy_file.html"

    raw_arr = uncompress()
        
    # write to numpy file
    if os.path.isfile(out_filepath):
        os.remove(out_filepath)

    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof: 
        t = time.time()

        np.save(out_filepath, raw_arr)

        print(f'time to save the array to numpy file: {time.time() - t}')
        visualize([prof, rprof, cprof], diagnostics_filepath)
        

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
    print(f'Time to write 2.5GB of generated data into npy stack: {t}')


def create_arrays():
    for array_config in test_arrays:
        file_path = get_file_path(array_config)

        create_random_cube(
            storage_type="hdf5",
            file_path=file_path,
            shape=(1400, 1400, 1400),  
            physik_chunks_shape=array_config['physik_cs'], 
            dtype=np.float16, 
            distrib=array_config['distrib'], 
            compression=array_config['compression']
        )

        try: # sanity check 
            with h5py.File(file_path, 'r') as f:
                if len(list(f.keys())) > 0:
                    dataset = f['data']

                print(f'dataset.shape: {dataset.shape}')
                print(f'dataset.chunks: {dataset.chunks}')
        except Exception as e:
            print(traceback.format_exc())
            print('Sanity check failed.')


def write_to_npy_stack(out_dir, arr):
    a__, b__, c__ = da.to_npy_stack(out_dir, arr, axis=0)
    _ = dask.base.compute_as_if_collection(a__, b__, c__)


def get_file_path(array_config):
    file_path = os.path.join(data_dir, array_config['name'] + 'hdf5')
    return file_path


def enable_optimization():
    manual_config_dask(buffer_size=buffer_size, opti=True, sched_opti=True)


def disable_optimization():
    manual_config_dask(buffer_size=buffer_size, opti=False, sched_opti=False)


def experiment2(data_dir, buffer_size, chunks_shape):
    """ Evaluation of read/write time from hdf5 file to npy/hdf5 file(s).

    Arguments:
    ----------
        data_dir: Directory in which to create and store input/output arrays.
        buffer_size: size of buffer for clustered strategy.
        chunks_shape: Shape of block to extract and save.
    """

    def get_array():
        """ Get a block of input array from hdf5 file.
        """
        t = time.time() 
        arr = get_dask_array_from_hdf5(file_path, logic_chunks_shape=chunks_shape)
        arr = arr[:chunks_shape[0], :chunks_shape[1], :chunks_shape[2]]
        print(f'time to load compressed file in cache: {time.time() - t}') 
        return arr

    def uncompress():
        """ Get data from hdf5 compressed file.
        """
        t = time.time()
        raw_arr = arr.compute()
        print(f'time to load data in RAM: {time.time() - t}')
        return raw_arr

    functions = [
        uncompress_to_npy,
        uncompress_to_nps,
        uncompress_to_hdf5,
        onthefly_to_nps,
        onthefly_to_hdf5
    ]    

    for array_config in test_arrays:
        file_path = get_file_path(array_config)
        print(f"Processing file: {file_path}...")

        with dask.config.set(scheduler='single-threaded'):
            for F in functions:
                flush_cache()
                arr = get_array(file_path)
                enable_optimization()
                F()
                disable_optimization()
    return


if __name__ == "__main__":
    experiment2(
        data_dir='data',
        buffer_size=5.5 * ONE_GIG,
        chunks_shape=(1400, 1400, 350)  
    )
        
    