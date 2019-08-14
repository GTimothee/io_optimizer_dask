import sys, os
import h5py
import optimize_io
from optimize_io.get_dicts import *

import tests_utils
from tests_utils import *

def test_get_arrays_dictionaries():
    graph = {'rechunk-merge-bcfb966a39aa5079f6457f1530dd85df': get_rechunk_dict_without_proxy_array_sample(),
             'rechunk-merge-a168f56ba79513b9ed87b2f22dd07458': get_rechunk_dict_from_proxy_array_sample(),
             'getitem-c6555b775be6a9d771866321a0d38252': get_getitem_dict_from_proxy_array_sample()}
    sample_file_path = 'tests/myfile.hdf5'
    f = h5py.File(sample_file_path,'w')
    dset = f.create_dataset("chunked", (100,100,100), chunks=(10,20,5))
    d = {
        'array-3ec4eddf5e385f67eb8007734372b503':{
            ('array-3ec4eddf5e385f67eb8007734372b503', 0, 0, 0): (None, 'array-original-68453165', (slice(1,1,1), slice(1,1,1), slice(1,1,1))),
            ('array-3ec4eddf5e385f67eb8007734372b503', 0, 0, 1): (None, 'array-original-68453165', (slice(1,1,1), slice(1,1,1), slice(1,1,1))),
            ('array-3ec4eddf5e385f67eb8007734372b503', 0, 0, 2): (None, 'array-original-68453165', (slice(1,1,1), slice(1,1,1), slice(1,1,1))),
            'array-original-68453165': dset
        },
        'array-6f870a321e8529128cb9bb82b8573db5':{
            ('array-6f870a321e8529128cb9bb82b8573db5', 0, 0, 0): (None, 'array-original-68453165', (slice(1,1,1), slice(1,1,1), slice(1,1,1))),
            ('array-6f870a321e8529128cb9bb82b8573db5', 0, 0, 1): (None, 'array-original-68453165', (slice(1,1,1), slice(1,1,1), slice(1,1,1))),
            ('array-6f870a321e8529128cb9bb82b8573db5', 0, 0, 2): (None, 'array-original-68453165', (slice(1,1,1), slice(1,1,1), slice(1,1,1))),
            'array-original-86453663': dset
        }
    }
    graph.update(d)
    slices_dict = {
        'array-3ec4eddf5e385f67eb8007734372b503': [(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,0,3),(0,1,3),(0,2,0),(0,2,1),(0,2,2)],
        'array-6f870a321e8529128cb9bb82b8573db5': [(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4),(0,0,5)]
    }
    array_to_original, original_array_chunks, original_array_shapes, original_array_blocks_shape = get_arrays_dictionaries(graph, slices_dict)
    f.close()
    os.remove(sample_file_path)
    expected_array_to_original = {
        'array-6f870a321e8529128cb9bb82b8573db5': 'array-original-86453663',
        'array-3ec4eddf5e385f67eb8007734372b503': 'array-original-68453165'
    }
    expected_array_shapes = {
        'array-original-86453663': (100,100,100),
        'array-original-68453165': (100,100,100)
    }
    expected_array_chunks = {
        'array-original-86453663': (10,20,5),
        'array-original-68453165': (10,20,5)
    }
    expected_blocks_shape = {
        'array-original-86453663': (10,5,20),
        'array-original-68453165': (10,5,20)
    }
    assert array_to_original == expected_array_to_original
    assert expected_array_chunks == original_array_chunks
    assert expected_array_shapes == original_array_shapes
    assert expected_blocks_shape == original_array_blocks_shape


def test_get_original_array_from_proxy_array_name():
    d = {
        'array-685431684531':{
            ('array-685431684531', 0, 0, 0): (None, 'array-original-68453165', (slice(1,1,1), slice(1,1,1), slice(1,1,1))),
            ('array-685431684531', 0, 0, 1): (None, 'array-original-68453165', (slice(1,1,1), slice(1,1,1), slice(1,1,1))),
            ('array-685431684531', 0, 0, 2): (None, 'array-original-68453165', (slice(1,1,1), slice(1,1,1), slice(1,1,1))),
            'array-original-68453165': 'original-array-obj'
        }
    }
    name, obj = get_original_array_from_proxy_array_name(graph=d, proxy_array_name='array-685431684531')
    assert name == 'array-original-68453165' and obj == 'original-array-obj'


def test_get_array_block_dims():
    shape = (500, 1200, 300)
    chunks = (100, 300, 20)
    block_dims = get_array_block_dims(shape, chunks)
    expected = (5, 4, 15)
    assert block_dims == expected