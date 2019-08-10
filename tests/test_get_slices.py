
from optimize_io.get_slices import *

import utils
from utils import *

import sys

# TODO: verify deps_dict for all tests


def test_BFS_connected_components():
    graph = {
        "a":["b", "c"],
        "b":["a", "d"],
        "c":["a"],
        "d":["b"],
        "e":["f", "g"],
        "f":["e"],
        "g":["e"]
    }

    expected_comps = {
        0: ["a", "b", "c", "d"],
        1: ["e", "f", "g"]
    }

    comps = BFS_connected_components(graph)

    for k, v in expected_comps.items():
        if k not in comps:
            print("error in", sys._getframe().f_code.co_name)
            print("key missing")
            return
        if set(comps[k]) != set(v):
            print("error in", sys._getframe().f_code.co_name)
            print("bad component")
            return 
    
    print("success")
            

def test_get_used_getitems_from_graph():
    data_path = '/home/user/Documents/workspace/projects/samActivities/experience3/tests/data/bbsamplesize.hdf5'
    key = "data"
    arr = get_dask_array_from_hdf5(data_path, key)
    case = 'slabs_dask_interpol'
    dask_array = logical_chunks_tests(arr, case, number_of_arrays=1)
    graph = dask_array.dask.dicts

    expected_length = 36
    used_getitems = get_used_getitems_from_graph(graph, undirected=False)
    used_getitems = list(set(used_getitems))

    if len(used_getitems) != expected_length:
        print("error in", sys._getframe().f_code.co_name)
        print("expected", expected_length, "\ngot", len(used_getitems))
        return

    print("success")


def test_get_getitems_from_graph():
    """ 
    """
    data_path = '/home/user/Documents/workspace/projects/samActivities/experience3/tests/data/bbsamplesize.hdf5'
    key = "data"
    arr = get_dask_array_from_hdf5(data_path, key)
    case = 'slabs_dask_interpol'
    dask_array = logical_chunks_tests(arr, case, number_of_arrays=1)
    graph = dask_array.dask.dicts

    expected_length = 245
    used_getitems = get_getitems_from_graph(graph)
    used_getitems = list(set(used_getitems))

    if len(used_getitems) != expected_length:
        print("error in", sys._getframe().f_code.co_name)
        print("expected", expected_length, "\ngot", len(used_getitems))
        return

    print("success")
  

def test_add_or_create_to_list_dict():
    d = {'a': [1], 'c': [5, 6]}
    d = add_or_create_to_list_dict(d, 'b', 2)
    d = add_or_create_to_list_dict(d, 'b', 3)
    d = add_or_create_to_list_dict(d, 'b', 4)
    expected = {'a': [1], 'b': [2, 3, 4], 'c': [5, 6]}
    
    if expected != d:
        print("error")
        print(d, "VS", expected)
        return
    print("success")


def test_get_keys_from_graph():
    d = {'a-165152': 1, 
         'b-865134': 2,
         'a-864531': 3,
         'c-864535': 4}
    keys_dict = get_keys_from_graph(d, printer=False)
    expected = {'a': ['a-165152', 'a-864531'],
                'b': ['b-865134'],
                'c': ['c-864535']}
    
    if not expected == keys_dict:
        print("error in", sys._getframe().f_code.co_name)
        print(expected, "VS", keys_dict)
        return 
    print("success")


def test_get_rechunk_subkeys():
    d = get_rechunk_dict_without_proxy_array_sample()
    split_keys, merge_keys = get_rechunk_subkeys(d)

    expected_merge_keys = [('rechunk-merge-bcfb966a39aa5079f6457f1530dd85df', 0, 0, 0),
                           ('rechunk-merge-bcfb966a39aa5079f6457f1530dd85df', 0, 0, 1)]

    expected_split_keys = [('rechunk-split-bcfb966a39aa5079f6457f1530dd85df', 0),
                           ('rechunk-split-bcfb966a39aa5079f6457f1530dd85df', 1),
                           ('rechunk-split-bcfb966a39aa5079f6457f1530dd85df', 2),
                           ('rechunk-split-bcfb966a39aa5079f6457f1530dd85df', 3),
                           ('rechunk-split-bcfb966a39aa5079f6457f1530dd85df', 4),
                           ('rechunk-split-bcfb966a39aa5079f6457f1530dd85df', 5)]

    for i, j in zip(split_keys, expected_split_keys):
        if i != j:
            print("error")
            print(i)
            print(j, '\n')
            return
                
    for i, j in zip(merge_keys, expected_merge_keys):
        if i != j:
            print("error")
            print(i)
            print(j, '\n')
            return
    print("success")


def test_test_source_key():
    slices_dict = dict()
    deps_dict = dict()
    sample_source_key = ['array-645318645', 
                         slice(None, None, None), 
                         slice(None, None, None), 
                         slice(None, None, None)]
    has_failed = [False, False]
    expected = {'array-645318645': [(sample_source_key[1], sample_source_key[2], sample_source_key[3])]}
    dep_key = 'sample_dependent_key'
    expected_deps = {'array-645318645': [dep_key]}
    slices_dict, deps_dict = test_source_key(slices_dict, deps_dict, tuple(sample_source_key), dep_key)
    sample_source_key[0] = "tmp"
    try:
        result, _ = test_source_key(dict(), dict(), tuple(sample_source_key), dep_key)
    except:
        has_failed[0] = True
    sample_source_key[0] = (864513, 86513, 4651, 3465)
    try:
        result, _ = test_source_key(dict(), dict(), tuple(sample_source_key), dep_key)
    except:
        has_failed[1] = True
    
    if has_failed != [False, True]:
        print("error in", sys._getframe().f_code.co_name, "(failure management)")
        print(has_failed)
        return

    for i, j in zip(expected, slices_dict):
        if i != j:
            print("error in", sys._getframe().f_code.co_name)
            print(i)
            print(j, '\n')
            return

    for k, v in expected_deps.items():
        if tuple(v) != tuple(deps_dict[k]):
            print("error in", sys._getframe().f_code.co_name)
            print(k)
            print(tuple(v))
            print(tuple(deps_dict[k]), '\n')
            return
    print("success")

    
def test_get_slices_from_rechunk_subkeys():
    rechunk_merge_graph = get_rechunk_dict_from_proxy_array_sample()
    split_keys, merge_keys = get_rechunk_subkeys(rechunk_merge_graph)
    slices_dict, deps_dict = get_slices_from_rechunk_subkeys(rechunk_merge_graph, split_keys, merge_keys)
    expected_name = 'array-3ec4eddf5e385f67eb8007734372b503'
    expected_list = [(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,0,3),(0,1,3),(0,2,0),(0,2,1),(0,2,2)]
    expected = {expected_name: expected_list}
    if not set(slices_dict) == set(expected):
        print("error in", sys._getframe().f_code.co_name)
        print(slices_dict)
        print(expected)
        return 
    print("success")


def test_get_slices_from_rechunk_keys():
    names = ['array-864513645',
             'array-8645346531',
             'array-6845312645']

    d = {'rechunk-merge-65312653120': get_rechunk_dict_from_proxy_array_sample(array_names=names, add_list=[1]),
         'rechunk-merge-64531264531': get_rechunk_dict_from_proxy_array_sample(array_names=names, add_list=[2]),
         'rechunk-merge-86453126543': get_rechunk_dict_from_proxy_array_sample(array_names=names, add_list=[3])}

    rechunk_keys = list(d.keys())

    slices_dict, deps_dict = get_slices_from_rechunk_keys(d, rechunk_keys)
    expected = dict()
    expected[names[0]] = [(0,0,3),(0,1,3)]
    expected[names[1]] = [(0,2,0)]
    expected[names[2]] = [(0,2,1),(0,2,2)]
    expected['array-3ec4eddf5e385f67eb8007734372b503'] = [(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2)]
    
    for k in list(expected.keys()):
        if set(expected[k]) != set(slices_dict[k]):
            print("error in", sys._getframe().f_code.co_name, "at", k)
            print(slices_dict)
            print(expected)
            return 
    print("success")


def test_get_slices_from_getitem_subkeys():
    getitem_graph = get_getitem_dict_from_proxy_array_sample()
    used_getitems = [('getitem-c6555b775be6a9d771866321a0d38252',0,0,0),
                     ('getitem-c6555b775be6a9d771866321a0d38252',0,0,1),
                     ('getitem-c6555b775be6a9d771866321a0d38252',0,0,2),
                     ('getitem-c6555b775be6a9d771866321a0d38252',0,0,3),
                     ('getitem-c6555b775be6a9d771866321a0d38252',0,0,4),
                     ('getitem-c6555b775be6a9d771866321a0d38252',0,0,5)]
    slices_dict, deps_dict = get_slices_from_getitem_subkeys(getitem_graph, used_getitems)
    expected_name = 'array-6f870a321e8529128cb9bb82b8573db5'
    expected = {expected_name: [(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4),(0,0,5)]}
    if not slices_dict == expected:
        print("error in", sys._getframe().f_code.co_name)
        print(slices_dict)
        print(expected)
        return 
    print("success")


def test_get_slices_from_getitem_keys():
    graph = get_graph_with_getitem()
    getitem_keys = ['getitem-430f856c4196ad50518e167d79ffd894']
    used_getitems = [('getitem-430f856c4196ad50518e167d79ffd894',0,0,0),
                     ('getitem-430f856c4196ad50518e167d79ffd894',0,0,1),
                     ('getitem-430f856c4196ad50518e167d79ffd894',0,0,3),
                     ('getitem-430f856c4196ad50518e167d79ffd894',0,0,4),
                     ('getitem-430f856c4196ad50518e167d79ffd894',0,0,5),
                     ('getitem-430f856c4196ad50518e167d79ffd894',0,0,6)]
    slices_dict, deps_dict = get_slices_from_getitem_keys(graph, getitem_keys, used_getitems)
    expected_name = 'array-4d8aa96f6f06806aeb9a11b75751b175'
    expected = {expected_name: [(0,0,0),(0,0,1),(0,0,3),(0,0,4),(0,0,5),(0,0,6)]}
    if not slices_dict == expected:
        print("error in", sys._getframe().f_code.co_name)
        print(slices_dict)
        print(expected)
        return 
    print("success")


def test_get_slices_from_dask_graph():
    graph = {'rechunk-merge-bcfb966a39aa5079f6457f1530dd85df': get_rechunk_dict_without_proxy_array_sample(),
             'rechunk-merge-a168f56ba79513b9ed87b2f22dd07458': get_rechunk_dict_from_proxy_array_sample(),
             'getitem-c6555b775be6a9d771866321a0d38252': get_getitem_dict_from_proxy_array_sample()}

    used_getitems = [('getitem-c6555b775be6a9d771866321a0d38252',0,0,0),
                     ('getitem-c6555b775be6a9d771866321a0d38252',0,0,1),
                     ('getitem-c6555b775be6a9d771866321a0d38252',0,0,2),
                     ('getitem-c6555b775be6a9d771866321a0d38252',0,0,3),
                     ('getitem-c6555b775be6a9d771866321a0d38252',0,0,5)]

    slices_dict, deps_dict = get_slices_from_dask_graph(graph, used_getitems)

    expected = {
        'array-3ec4eddf5e385f67eb8007734372b503': [(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,0,3),(0,1,3),(0,2,0),(0,2,1),(0,2,2)],
        'array-6f870a321e8529128cb9bb82b8573db5': [(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,5)]
    }

    for key, val in expected.items():
        if set(slices_dict[key]) != set(val):
            print("error in", sys._getframe().f_code.co_name)
            print(key)
            print(set(slices_dict[key]))
            print(set(val))
            return 
    print("success")