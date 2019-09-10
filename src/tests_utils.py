""" A list of utility functions for the tests
"""

import math
import os
import dask_utils_perso
from dask_utils_perso.utils import (create_random_cube, load_array_parts,
    get_dask_array_from_hdf5)

ONE_GIG = 1000000000

__all__ = ['ONE_GIG',
           'flush_cache',
           'get_arr_shapes',
           'get_test_arr',
           'neat_print_graph']

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



def get_arr_list():
    """ Return a list of dask array. Each dask array being a block of the input array.
    """
    _, ncs, dims = get_arr_shapes(arr)
    arr_list = list()
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if len(arr_list) < nb_chunks:
                    arr_list.append(load_array_parts(arr=arr,
                                                     geometry="right_cuboid",
                                                     shape=ncs,
                                                     upper_corner=(i * ncs[0],
                                                                   j * ncs[1],
                                                                   k * ncs[2]),
                                                     random=False))
    return arr_list


def sum_chunks(arr, nb_chunks):
    """ Sum chunks together.

    Arguments:
        arr: array from which blocks will be sum
        nb_chunks: number of chunks to sum
    """
    arr_list = get_arr_list(arr)
    sum_arr = arr_list.pop(0)
    for a in arr_list:
        sum_arr = sum_arr + a
    return sum_arr
    

def get_test_arr(file_path, chunk_shape=None, shape=None, test_case=None, nb_chunks=2, overwrite=False, split_file=None):
    """ Load or create Dask Array for tests. You can specify a test case too.

    If file exists the function returns the array.
    If chunk_shape given the function rechunk the array before returning it.
    If file does not exist it will be created using "shape" parameter.
    If file does exist and chunk_shape different than 

    Arguments:
        file_path: File containing the array, will be created if does not exist.
        chunk_shape: Shape of the array to load.
        shape: Shape of the array to create if does not exist.
        test_case: Test case. If None, returns the test array.
        nb_chunks: Number of chunks to treat in the test case.
        overwrite: Use the chunk_shape to create a new array, overwriting if file_path already used.
        split_file: for the test case 'split'
    """

    def get_or_create_array():
        arr = None 
    
        if overwrite and shape and os.path.isfile(file_path):
            os.remove(file_path)

        if not os.path.isfile(file_path):
            if not shape: 
                raise ValueError("No shape to create the array")

            if file_path.split('.')[-1] == 'hdf5':
                dask_utils_perso.utils.create_random_cube(storage_type="hdf5",
                                                        file_path=file_path,
                                                        shape=shape,
                                                        chunks_shape=None,
                                                        dtype="float16")
            else:
                raise ValueError("File format not supported yet.")
        try:
            arr = get_dask_array_from_hdf5(file_path, key='data')
        except: 
            raise IOError("failed to load file")
        
        return arr

    arr = get_or_create_array()

    if chunk_shape:
        arr = arr.rechunk(chunk_shape)

    if test_case:
        if test_case == 'sum':
            arr = sum_chunks(arr, nb_chunks=nb_chunks)
        elif test_case == 'split':
            arr = split_array(arr, split_file)
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