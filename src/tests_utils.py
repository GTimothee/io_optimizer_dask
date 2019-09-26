""" A list of utility functions for the tests
"""
import dask
import math
import os
import dask_utils_perso
from dask_utils_perso.utils import (create_random_cube, load_array_parts,
    get_dask_array_from_hdf5)


# shapes used for the first experiment 
CHUNK_SHAPES_EXP1 = {'slabs_dask_interpol': ('auto', (1210), (1400)), 
                    'slabs_previous_exp': (7, (1210), (1400)),
                    'blocks_dask_interpol': (220, 242, 200), 
                    'blocks_previous_exp': (770, 605, 700)}
ONE_GIG = 1000000000
SUB_BIGBRAIN_SHAPE = (1540, 1610, 1400)


def configure_dask(config, optimize_func=None):
    """ Apply configuration to dask to parameterize the optimization function.
    """
    if not config:
        raise ValueError("Empty configuration object.")

    opti_funcs = [] if config.opti else list()
    scheduler_opti = config.scheduler_opti if config.opti else False
    dask.config.set({'optimizations': opti_funcs,
                     'io-optimizer': {
                        'memory_available': config.buffer_size,
                        'scheduler_opti': scheduler_opti}
                    })


class CaseConfig():
    """ Contains the configuration for a test.
    """
    def __init__(self, array_filepath, chunks_shape):
        self.array_filepath = array_filepath
        self.chunks_shape = chunks_shape

    def optimization(self, opti, scheduler_opti, buffer_size):
        self.opti = opti 
        self.scheduler_opti = scheduler_opti
        self.buffer_size = buffer_size

    def sum_case(self, nb_chunks):
        self.test_case = 'sum'
        self.nb_chunks = nb_chunks

    def split_case(self, in_filepath, out_filepath):
        self.test_case = 'split'
        self.in_filepath = in_filepath       
        if os.path.isfile(out_filepath):
            os.remove(out_filepath)
        self.out_file = h5py.File(self.out_filepath, 'w') 

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
        arr: dask array
        nb_chunks: if None then function returns all arrays, else function returns n=nb_chunks arrays
    """
    _, chunk_shape, dims = get_arr_shapes(arr)
    arr_list = list()
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
    return arr_list


def sum_chunks(arr, nb_chunks):
    """ Sum chunks together.

    Arguments:
        arr: array from which blocks will be sum
        nb_chunks: number of chunks to sum
    """
    
    arr_list = get_arr_list(arr, nb_chunks)
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
        file_path = config.array_filepath
        arr = None 

        if not os.path.isfile(file_path):
            print("File not found, attempting to create the array...")
            if not config.shape: 
                raise ValueError("No shape to create the array")

            if file_path.split('.')[-1] == 'hdf5':
                dask_utils_perso.utils.create_random_cube(storage_type="hdf5",
                                                        file_path=file_path,
                                                        shape=config.chunks_shape,  # TODO: change this to None or add physical rechunk to config
                                                        chunks_shape=None,  # this chunk shape is for physical chunks
                                                        dtype="float16")
            else:
                raise ValueError("File format not supported yet.")
        
        if config.chunks_shape:
            arr = get_dask_array_from_hdf5(file_path, logic_chunks_shape=config.chunks_shape)
        else:
            arr = get_dask_array_from_hdf5(file_path)
        return arr

    arr = get_or_create_array(config)
    if config.chunks_shape and arr.chunks != config.chunks_shape:  # to be removed, normally should automatically get good schunk
        arr = arr.rechunk(config.chunks_shape)

    case = getattr(config, 'test_case', None)
    if case:
        if case == 'sum':
            arr = sum_chunks(arr, config.nb_chunks)
        elif case == 'split':
            arr = split_array(arr, config.out_file)
    return arr


def split_array(arr, f):
    """ Split an array given its chunk shape. Output is a hdf5 file with as many datasets as chunks.
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