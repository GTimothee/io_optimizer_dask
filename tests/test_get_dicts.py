import sys
sys.path.insert(2,'/home/user/Documents/workspace/projects/samActivities/tests/')
import optimize_io
from optimize_io import get_dicts 
from get_dicts import *
import h5py

import utils
from utils import *

def test_get_arrays_dictionaries():
    graph = {'rechunk-merge-bcfb966a39aa5079f6457f1530dd85df': get_rechunk_dict_without_proxy_array_sample(),
             'rechunk-merge-a168f56ba79513b9ed87b2f22dd07458': get_rechunk_dict_from_proxy_array_sample(),
             'getitem-c6555b775be6a9d771866321a0d38252': get_getitem_dict_from_proxy_array_sample()}
    f = h5py.File('myfile.hdf5','w')
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
    if array_to_original != expected_array_to_original:
        print("error in", sys._getframe().f_code.co_name)
        print(array_to_original)
        return
    elif expected_array_chunks != original_array_chunks:
        print("error in", sys._getframe().f_code.co_name)
        print(original_array_chunks)
        return
    elif expected_array_shapes != original_array_shapes:
        print("error in", sys._getframe().f_code.co_name)
        print(original_array_shapes)
        return
    elif expected_blocks_shape != original_array_blocks_shape:
        print("error in", sys._getframe().f_code.co_name)
        print(original_array_blocks_shape)    
        return
    print('success')


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
    if name != 'array-original-68453165' or obj != 'original-array-obj':
        print("error in", sys._getframe().f_code.co_name)
        print(name, obj)
        return 
    print("success")


def test_get_array_block_dims():
    shape = (500, 1200, 300)
    chunks = (100, 300, 20)
    block_dims = get_array_block_dims(shape, chunks)
    expected = (5, 4, 15)
    if block_dims != expected:
        print("error in", sys._getframe().f_code.co_name)
        print("expected", expected, ", got", block_dims)
        return 
    print("success")