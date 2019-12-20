""" A list of utility functions for the tests

to be renamed "util.py"
"""
import dask
import dask.array as da
import math
import os
import h5py
import datetime
import csv

import dask_utils_perso
from dask_utils_perso.utils import get_random_slab, get_random_cubic_block, get_random_right_cuboid


ONE_GIG = 1000000000
SUB_BIGBRAIN_SHAPE = (1540, 1610, 1400)
LOG_TIME = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())
CHUNK_SHAPES_EXP1 = {'slabs_dask_interpol': ('auto', (1210), (1400)), 
                    'slabs_previous_exp': (7, (1210), (1400)),
                    'blocks_dask_interpol': (220, 242, 200), 
                    'blocks_previous_exp': (770, 605, 700)}


def get_dask_array_from_hdf5(file_path="tests/data/sample.hdf5", cast=True, key='/data', logic_chunks_shape="auto"):
    """
    Arguments:
    ----------
        file path: path to hdf5 file (string)
        key: key of the dictionary to retrieve data
        cast: if you want to cast the dataset into a dask array.
            If you want to do it yourself (ex for adjusting the chunks),
            set this parameter to False.
    """
    f = h5py.File(file_path, 'r')
    if len(list(f.keys())) > 0:
        if not cast:
            return f[key]
        dataset = f[key]
        if dataset.chunks:  # if dataset is chunked use the same logical chunks shape as physical chunks shape
            return da.from_array(dataset, chunks=dataset.chunks)
        else:  # if no physical chunked then should choose a chunks shape, "auto" is automatic ~100MB chunk size
            return da.from_array(dataset, chunks=logic_chunks_shape)
    else:
        print('Key not found. Aborting.')


def load_array_parts(arr, geometry="slabs", nb_elements=0, shape=None, axis=0, random=True, seed=0, upper_corner=(0,0,0), as_numpy=False):
    """ Load part of array.
    Load 1 (or more parts -> one for the moment) of a too-big-for-memory array from file into memory.
    -given 1 or more parts (not necessarily close to each other)
    -take into account geometry
    -take into account storage type (unique_file or multiple_files) thanks to dask

    Arguments:
    ----------
        geometry: name of geometry to use
        nb_elements: nb elements wanted, not used for right_cuboid geometry
        shape: shape to use for right_cuboid
        axis: support axis for the slab
        random: if random cut or at a precise position. If set to False, upper_corner should be set.
        upper_corner: upper corner of the block/slab to be extracted (position from which to extract in the array).

    Returns a numpy array
    """
    if geometry not in ["slabs", "cubic_blocks", "right_cuboid"]:
        print("bad geometry type. Aborting.")
        return

    if geometry == "slabs":
        if random == False and len(upper_corner) != 2:
            print("Bad shape for upper corner: must be of dimension 2. Aborting.")
            return
        arr = get_random_slab(nb_elements, arr, axis, seed, random, pos=upper_corner)
    elif geometry == "cubic_blocks":
        arr = get_random_cubic_block(nb_elements, arr, seed, random, corner_index=upper_corner)
    elif geometry == "right_cuboid":
        arr = get_random_right_cuboid(arr, shape, seed, random, pos=upper_corner)

    if as_numpy:
        return arr.compute()
    else:
        return arr


def create_csv_file(filepath, columns, delimiter=',', mode='w+'):
    """
    args:
        file_path: path of the file to be created
        columns: column names for the prefix
        delimiter
        mode

    returns:
        csv_out: file handler in order to close the file later in the program
        writer: writer to write rows into the csv file
    """
    try:
        csv_out = open(filepath, mode=mode)
        writer = csv.writer(csv_out, delimiter=delimiter)
        writer.writerow(columns)
        return csv_out, writer
    except OSError:
        print("An error occured while attempting to create/write csv file.")
        exit(1)


def create_random_cube(storage_type, file_path, shape, axis=0, physik_chunks_shape=None, dtype=None, distrib='uniform', compression=None):
    """ Generate random cubic array from normal distribution and store it on disk.
    Pass a dtype to cast the input dask array.

    Arguments:
        storage_type: string
        shape: tuple or integer
        file_path: has to contain the filename if hdf5 type, should not contain a filename if npy type.
        axis: for numpy: splitting dimension because numpy store matrices
        physik_chunks_shape: for hdf5 only (as far as i am concerned for the moment)

    Example usage:
    >> create_random_cube(storage_type="hdf5",
                   file_path="tests/data/bbsamplesize.hdf5",
                   shape=(1540,1210,1400),
                   physik_chunks_shape=None,
                   dtype="float16")
    """
    if distrib == 'normal':
        print('Drawing from normal distribution')
        arr = da.random.normal(size=shape)
    else:
        print('Drawing from uniform distribution')
        arr = da.random.random(size=shape)
    if dtype:
        arr = arr.astype(dtype)
    save_arr(arr, storage_type, file_path, key='/data', axis=axis, chunks_shape=physik_chunks_shape, compression=compression)


def save_arr(arr, storage_type, file_path, key='/data', axis=0, chunks_shape=None, compression=None):
    """ Save dask array to hdf5 dataset or numpy file stack.
    """

    if storage_type == "hdf5":
        if chunks_shape:
            print(f'Using chunk shape {chunks_shape}')
            da.to_hdf5(file_path, key, arr, chunks=chunks_shape)
        else:
            if compression == "gzip":
                print('Using gzip compression')
                da.to_hdf5(file_path, key, arr, chunks=None, compression="gzip")
            else:
                print('Without compression')
                da.to_hdf5(file_path, key, arr, chunks=None)
    elif storage_type == "numpy":
        da.to_npy_stack(os.path.join(file_path, "npy/"), arr, axis=axis)


def get_dask_array_chunks_shape(dask_array):
    t = dask_array.chunks
    cs = list()
    for tupl in t:
        cs.append(tupl[0])
    return tuple(cs)


def configure_dask(config):
    """ Apply configuration to dask to parameterize the optimization function.

    Arguments:
    ---------
        config: A CaseConfig object.
    """
    if not config:
        raise ValueError("Empty configuration object.")
    manual_config_dask(config.buffer_size, config.opti, sched_opti=scheduler_opti)


def manual_config_dask(buffer_size=ONE_GIG, opti=True, sched_opti=True):
    """ Manual configuration of Dask as opposed to using CaseConfig object.

    Arguments:
    ----------
        buffer_size:
        opti:
        sched_opti:
    """

    print('Task graph optimization enabled:', opti)
    print('Scheduler optimization enabled:', sched_opti)

    opti_funcs = [optimize_func] if opti else list()
    dask.config.set({
        'optimizations': opti_funcs,
        'io-optimizer': {
            'memory_available': buffer_size,
            'scheduler_opti': sched_opti
        }
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

    def split_case(self, in_filepath, out_filepath, nb_blocks=None):
        """
        nb_blocks: nb_blocks to extract from the original array
        """
        self.test_case = 'split'
        self.in_filepath = in_filepath  # TODO: remove this we already have it as array_filepath
        self.nb_blocks = nb_blocks if nb_blocks else None
        if os.path.isfile(out_filepath):
            os.remove(out_filepath)
        self.out_filepath = out_filepath
        # print("split file path stored in config:", self.out_filepath)
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
        else:
            raise ValueError("Unsupported test case.")
        writer.writerow(data)


def flush_cache():
    os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches')   


def get_arr_shapes(arr):
    """ Routine that returns shape information on from dask array.

    Arguments:
    ----------
        arr: dask array

    Returns:
    --------
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
    ----------
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
    ----------
        arr: array from which blocks will be sum
        nb_chunks: number of chunks to sum
    """
    
    arr_list = get_arr_list(arr, nb_chunks)
    sum_arr = arr_list.pop(0)
    for a in arr_list:
        sum_arr = sum_arr + a
    return sum_arr


def get_or_create_array(config, npy_stack_dir=None):
    """ Load or create Dask Array for tests. You can specify a test case too.

    If file exists the function returns the array.
    If chunk_shape given the function rechunk the array before returning it.
    If file does not exist it will be created using "shape" parameter.

    Arguments (from config object):
    ----------
        file_path: File containing the array, will be created if does not exist.
        chunk_shape: 
        shape: Shape of the array to create if does not exist.
        test_case: Test case. If None, returns the test array.
        nb_chunks: Number of chunks to treat in the test case.
        overwrite: Use the chunk_shape to create a new array, overwriting if file_path already used.
        split_file: for the test case 'split'
    """

    file_path = config.array_filepath
    if not os.path.isfile(file_path):
        raise FileNotFoundError()
    
    # get the file and rechunk logically using a chosen chunk shape, or dask default
    if npy_stack_dir:
        arr = da.from_npy_stack(dirname=npy_stack_dir, mmap_mode=None)
    else:
        if config.chunks_shape:
            arr = get_dask_array_from_hdf5(file_path, logic_chunks_shape=config.chunks_shape)
        else:
            arr = get_dask_array_from_hdf5(file_path) # TODO: see what happens
    return arr


def create_random_array():
    file_path = '../' + os.path.join(os.getenv('DATA_PATH'), 'sample_array_nochunk.hdf5')
    create_random_cube(storage_type="hdf5",
        file_path=file_path,
        shape=(1540, 1210, 1400),
        physik_chunks_shape=None,
        dtype="float16")


def get_test_arr(config, npy_stack_dir=None):

    # create the dask array from input file path
    arr = get_or_create_array(config, npy_stack_dir=npy_stack_dir)
    
    # do dask arrays operations for the chosen test case
    case = getattr(config, 'test_case', None)
    # print("case in config", case)
    if case:
        if case == 'sum':
            arr = sum_chunks(arr, config.nb_chunks)
        elif case == 'split':
            arr = split_array(arr, config.out_file, config.nb_blocks)
    return arr


def split_array(arr, f, nb_blocks=None):
    """ Split an array given its chunk shape. Output is a hdf5 file with as many datasets as chunks.
    
    Arguments:
    ----------
        nb_blocks: nb blocks we want to extract
    """
    # arr_list = get_arr_list(arr, nb_blocks)
    datasets = list()

    # for hdf5:
    """for i, a in enumerate(arr_list):
        # print("creating dataset in split file -> dataset path: ", '/data' + str(i))
        # print("storing data of shape", a.shape)
        datasets.append(f.create_dataset('/data' + str(i), shape=a.shape))
    return da.store(arr_list, datasets, compute=False)"""

    # for numpy storage
    return da.to_npy_stack('data/numpy_data', arr, axis=0)


def neat_print_graph(graph):
    for k, v in graph.items():
        print("\nkey", k)
        if isinstance(v, dict):
            for k2, v2 in v.items():
                print("\nk", k2)
                print(v2, "\n")