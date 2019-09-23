""" A list of utility functions for the tests
"""

import math
import os
import dask_utils_perso
from dask_utils_perso.utils import (create_random_cube, load_array_parts,
    get_dask_array_from_hdf5)

ONE_GIG = 1000000000

__all__ = ['CaseConfig',
           'ONE_GIG',
           'flush_cache',
           'get_arr_shapes',
           'get_test_arr',
           'neat_print_graph']


SUB_BIGBRAIN_SHAPE = (1540, 1610, 1400)

class CaseConfig():
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


def flush_cache():
    os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')   


def get_arr_shapes(arr):
    """ Routine that returns shape information on from dask array.

    Arguments:
        arr: dask array
    Returns:
        shape: shape of the dask array
        chunks: shape of one chunk
        chunk_dims: number of chunks in eah dimension
    """
    shape = arr.shape
    chunks = tuple([c[0] for c in arr.chunks])
    chunk_dims = [len(c) for c in arr.chunks]  
    return shape, chunks, chunk_dims


def get_arr_list(arr, nb_chunks=None):
    """ Return a list of dask arrays. Each dask array being a block of the input array.
    Arguments:
        arr: original array of type dask array
        nb_chunks: if None then function returns all arrays, else function returns n=nb_chunks arrays
    """
    _, chunk_shape, dims = get_arr_shapes(arr)
    arr_list = list()
    arr_list_indices = list()
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if (nb_chunks and len(arr_list) < nb_chunks) or not nb_chunks:
                    upper_corner = (i * chunk_shape[0],
                                    j * chunk_shape[1],
                                    k * chunk_shape[2])
                    arr_list.append(load_array_parts(arr=arr,
                                                     geometry="right_cuboid",
                                                     shape=chunk_shape,
                                                     upper_corner=upper_corner,
                                                     random=False))
                    arr_list_indices.append((i, j, k))

    print("arr_list_indices", arr_list_indices)
    return arr_list


def sum_chunks(arr, nb_chunks):
    """ Sum chunks together.

    Arguments:
        arr: array from which blocks will be sum
        nb_chunks: number of chunks to sum
    """
    
    arr_list = get_arr_list(arr, nb_chunks)
    print("arr_list size", len(arr_list))
    sum_arr = arr_list.pop(0)
    for a in arr_list:
        sum_arr = sum_arr + a
    return sum_arr
    

def get_test_arr(config):
    """ Load or create Dask Array for tests. You can specify a test case too.

    If file exists the function returns the array.
    If chunk_shape given the function rechunk the array before returning it.
    If file does not exist it will be created using "shape" parameter.
    If file does exist and chunk_shape different than 

    Arguments (from config object):
        file_path: File containing the array, will be created if does not exist.
        chunk_shape: 
        shape: Shape of the array to create if does not exist.
        test_case: Test case. If None, returns the test array.
        nb_chunks: Number of chunks to treat in the test case.
        overwrite: Use the chunk_shape to create a new array, overwriting if file_path already used.
        split_file: for the test case 'split'
    """

    def get_or_create_array(config):
        file_path = config.input_file_path
        arr = None 
        if config.overwrite and config.shape and os.path.isfile(file_path):
            os.remove(file_path)

        if not os.path.isfile(file_path):
            if not config.shape: 
                raise ValueError("File not found, attempting to create the array... No shape to create the array")

            if file_path.split('.')[-1] == 'hdf5':
                dask_utils_perso.utils.create_random_cube(storage_type="hdf5",
                                                        file_path=file_path,
                                                        shape=config.shape,
                                                        chunks_shape=None,  # this chunk shape is for physical chunks
                                                        dtype="float16")
            else:
                raise ValueError("File format not supported yet.")
        try:
            arr = get_dask_array_from_hdf5(file_path, key='data')
        except: 
            raise IOError("failed to load file")
        
        return arr

    arr = get_or_create_array(config)

    print("original array shape", arr.shape)
    print("desired chunk_shape", config.chunk_shape)

    if config.chunk_shape:
        arr = arr.rechunk(config.chunk_shape)

    if config.test_case:
        if config.test_case == 'sum':
            arr = sum_chunks(arr, nb_chunks=config.nb_chunks)
        elif config.test_case == 'split':
            arr = split_array(arr, config.split_file)
    return arr


def split_array(arr, f):
    """
    output_file_path: an hdf5 file representing the splitted output: each dataset being an extracted chunk
    """
    arr_list = get_arr_list(arr)
    datasets = list()
    for i, a in enumerate(arr_list):
        datasets.append(f.create_dataset('/data' + str(i), shape=a.shape))
    return da.store(arr_list, datasets, compute=False)


def neat_print_graph(graph):
    for k, v in graph.items():
        print("\nkey", k)
        if isinstance(v, dict):
            for k2, v2 in v.items():
                print("\nk", k2)
                print(v2, "\n")