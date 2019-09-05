import sys, os, time, csv, h5py, copy
import numpy as np

import dask
import dask.array as da

import optimize_io
from optimize_io.main import optimize_func

import tests_utils
from tests_utils import *


cube_shapes = {
        "small": (1400, 1400, 1400), 
        "big": (3500, 3500, 3500),
    }


cube_refs = {
    'small': {
        True: 0,
        False: 1 
    },
    'big': {
        True: 2,
        False: 3 
    }
}

def load_json(file_path):
    import json
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def create_cube(data_path, ref, exp_id, chunked):
    file_path = os.path.join(data_path, str(ref) + '.hdf5')
    if not os.path.isfile(file_path):
        auto_chunk = True if chunked else None 
        create_random_cube(storage_type="hdf5",
                    file_path=file_path,
                    shape=cube_shapes[exp_id],
                    chunks_shape=auto_chunk,
                    dtype="float16")
    return file_path


def split():
    return


def merge():
    return


def main():
    ssd_path = os.getenv('SSD_PATH')
    hdd_path = os.getenv('HDD_PATH')
    workspace = os.getenv('BENCHMARK_DIR')
    chunk_shapes = load_json(os.path.join(workspace, 'chunk_shapes.json'))

    for exp_id in ['small']: # ['small', 'big']
        for data_path in [ssd_path]: # [ssd_path, hdd_path]
            hardware = "SSD" if data_path == ssd_path else "HDD"
            print("Running experiment on", hardware)

            for chunked in [False, True]:
                # create cube if does not exist
                ref = cube_refs[exp_id][chunked]
                file_path = create_cube(data_path, ref, exp_id, chunked)

                # do tests on cube for exp of type 'exp_id'
                for chunk_type in ['blocks', 'slabs']:
                    for shape in chunk_shapes[exp_id][chunk_type]:
                        print(shape)

                        # reshape if needed

                        # split

                        # merge