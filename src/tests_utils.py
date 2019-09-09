import math
import os
import dask_utils_perso
from dask_utils_perso.utils import (create_random_cube, load_array_parts,
    get_dask_array_from_hdf5)

""" A list of utility functions for the tests
"""

ONE_GIG = 1000000000


chunk_shapes = ['slabs_dask_interpol', 'slabs_previous_exp',
                'blocks_dask_interpol', 'blocks_previous_exp']


def get_arr_shapes(arr):
    shape = arr.shape
    chunks = tuple([c[0] for c in arr.chunks])  # shape of 1 chunk
    blocks_dims = [len(c) for c in arr.chunks]  # nb blocks in each dimension
    return shape, chunks, blocks_dims


def get_test_arr(case=None, nb_arr=2):
    """ Use this function to get array for tests
    if no parameters are given, function returns the default dask array
    """
    # load array
    data_path = get_test_array()
    key = 'data'
    arr = get_dask_array_from_hdf5(data_path, key)

    # create case
    if case and case == 'sum':
        chunk_shape = 'blocks_dask_interpol'
        arr = add_chunks(arr, chunk_shape, number_of_arrays=nb_arr)
        arr = arr.sum()

    return arr

def get_test_array(data_dir=None, shape=(1540, 1210, 1400), file_name='sample_array.hdf5', overwrite=False):
    """ Create data for the test if not created.
    """
    if not data_dir:
        data_dir = os.environ.get('DATA_PATH')
    data_path = os.path.join(data_dir, file_name)

    if os.path.isfile(data_path) and overwrite:
        os.remove(data_path)

    if not os.path.isfile(data_path):
        dask_utils_perso.utils.create_random_cube(storage_type="hdf5",
                                                  file_path=data_path,
                                                  shape=shape,
                                                  chunks_shape=None,
                                                  dtype="float16")
    return data_path


def add_chunks(arr, case, number_of_arrays):
    if case == 'slabs_dask_interpol':
        return arr.rechunk(('auto', (1210), (1400)))
    elif case == 'slabs_previous_exp':
        return arr.rechunk((7, (1210), (1400)))
    elif case == 'blocks_dask_interpol':
        return arr
    elif case == 'blocks_previous_exp':
        return arr.rechunk((770, 605, 700))
    else:
        raise ValueError("error")
    
    print("new chunks", arr.shape, arr.chunks)
    print("new_chunks_shape", new_chunks_shape)

    ncs = new_chunks_shape
    all_arrays = list()
    for i in range(nb_chunks[0]):
        for j in range(nb_chunks[1]):
            for k in range(nb_chunks[2]):
                all_arrays.append(load_array_parts(arr=arr,
                                                   geometry="right_cuboid",
                                                   shape=new_chunks_shape,
                                                   upper_corner=(i * ncs[0],
                                                                 j * ncs[1],
                                                                 k * ncs[2]),
                                                   random=False))

    # to del:
    all_arrays = all_arrays[:number_of_arrays]
    a5 = all_arrays.pop(0)
    for a in all_arrays:
        a5 = a5 + a
    return a5


def neat_print_graph(graph):
    for k, v in graph.items():
        print("\nkey", k)
        if isinstance(v, dict):
            for k2, v2 in v.items():
                print("\nk", k2)
                print(v2, "\n")