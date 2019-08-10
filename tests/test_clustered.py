
from optimize_io.clustered import *


import utils
from utils import *

import sys


def test_get_buffer_slices_from_original_array():
    load = [4, 5, 6, 7, 8] # TODO verify que cest bien un block (normalement oui)
    shape = (5, 3, 2)
    original_array_chunk = (10, 20, 30)

    """
    4e = (0, 2, 0) 
    8e = (1, 1, 0)  la fin du 8e est le debut du 9e => 9e = (1, 1, 1)

    mins= 0, 1, 0 => (0, 20, 0)
    maxs = 1, 2, 1 => (10, 40, 30)
    d'où slices = (0, 10, None), (20, 40, None), (0, 30, None)
    """

    expected_slices = (slice(0, 20, None), slice(0, 60, None), slice(0, 60, None))
    slices = get_buffer_slices_from_original_array(load, shape, original_array_chunk)
    if slices != expected_slices:
        print("error in", sys._getframe().f_code.co_name)
        print("got", slices, ", expected", expected_slices)    
        return
    print("success")


def test_convert_proxy_to_buffer_slices():
    proxy_key = ("array-645364531", 1, 1, 0)
    merged_task_name = "buffer-645136513-5-13"
    slices = (slice(5, 10, None), slice(10, 20, None), slice(None, None, None))
    array_to_original = {"array-645364531": "array-original-645364531"}
    original_array_chunks = {"array-original-645364531": (10, 20, 30)}
    original_array_blocks_shape = {"array-original-645364531": (5, 3, 2)}
    pos_in_buffer, result_slices = convert_proxy_to_buffer_slices(proxy_key, merged_task_name, slices, array_to_original, original_array_chunks, original_array_blocks_shape)
    
    """
    sizeofslice = 6
    sizeof_row = 2
    pos in img = 1*6 + 1*2 + 0*1 = 8
    pos in buffer = 8 - 5 = 3   
    pos in buffer in 3d = (0, 1, 1) => 0,20,30
    slices in proxy: 5,10  10,20  all
    5,10  30,40  30,60 
    """

    expected_slices = (slice(5, 10, None), slice(30, 40, None), slice(30, 60, None))
    if result_slices != expected_slices:
        print("error in", sys._getframe().f_code.co_name)
        print("got", result_slices, ", expected", expected_slices)    
        return
    print('success')


def test_add_getitem_task_in_graph():
    graph = dict()
    buffer_node_name = 'buffer-465316453-10-23'
    array_proxy_key = ("array-645364531", 3, 1, 2)
    array_to_original = {"array-645364531": "array-original-645364531"}
    original_array_chunks = {"array-original-645364531": (10,20,30)}
    original_array_blocks_shape = {"array-original-645364531": (5, 3, 2)}

    """
    size_of_slice = 6
    size_of_row = 2 
    pos in img = 3*6 + 1*2 + 2*1 = 22
    pos in buffer = 22 - 10 = 12
    pos in buffer in 3d = (2, 0, 0)
    pos in buffer not in terms of block = (20:30, 0:20, 0:30)
    """

    getitem_task_name, graph = add_getitem_task_in_graph(graph, buffer_node_name, array_proxy_key, array_to_original, original_array_chunks, original_array_blocks_shape)
    expected_slices = (slice(20, 30, None), slice(0, 20, None), slice(0, 30, None))

    # try retrieving data from graph
    try:
        buffer_proxy_key = list(graph.keys())[0]
        buffer_proxy_dict = graph[buffer_proxy_key]
        buffer_proxy_subtask_key = list(buffer_proxy_dict.keys())[0]
        buffer_proxy_subtask_val = buffer_proxy_dict[buffer_proxy_subtask_key]
        
        buffer_proxy_name = buffer_proxy_subtask_key[0]
        pos_in_buffer = buffer_proxy_subtask_key[1:]
        getitem, buffer_key, slices_from_buffer = buffer_proxy_subtask_val
    except:
        print("error in", sys._getframe().f_code.co_name)
        print("try failed")
        print(graph)
        return

    # if expected data is here, test the values
    error = False
    if pos_in_buffer != (2, 0, 0):
        error = True
    if buffer_key[0] != 'buffer-465316453-10-23':
        error = True
    if buffer_proxy_name != 'buffer-465316453-10-23-proxy':
        error = True
    if slices_from_buffer != expected_slices:
        error = True
    if error:
        print("error in", sys._getframe().f_code.co_name)
        print("checks failed")
        print("pos_in_buffer", pos_in_buffer)
        print("buffer_key[0]", buffer_key[0])
        print("buffer_proxy_name", buffer_proxy_name)
        print("slices_from_buffer", slices_from_buffer)
        return
    print('success')


def test_recursive_search_and_update():
    graph = dict()
    buffer_node_name = 'buffer-465316453-10-23'
    array_to_original = {"array-645364531": "array-original-645364531"}
    original_array_chunks = {"array-original-645364531": (10, 20, 30)}
    original_array_blocks_shape = {"array-original-645364531": (5, 3, 2)}

    _list = [[[('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 1), ("array-645364531", 2, 0, 0)],
              [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 2), ("array-645364531", 1, 2, 0)],
              [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 3), ("array-645364531", 2, 2, 2)]],
            [[('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 4), ("array-645364531", 2, 1, 1)],
             [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 5), ("array-645364531", 3, 1, 1)],
             [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 6), ("array-645364531", 3, 1, 2)]]]


    load = [(2,0,0),(1,2,0),(2,2,2),(2,1,1),(3,1,1),(3,1,2)]
    shape = original_array_blocks_shape["array-original-645364531"]
    load = [_3d_to_numeric_pos(l, shape, order='C') for l in load]
    load = sorted(load)


    graph, _list = recursive_search_and_update(graph, load, _list, buffer_node_name, array_to_original, original_array_chunks, original_array_blocks_shape)
    

    slices_list = [("array-645364531", 2, 0, 0), ("array-645364531", 1, 2, 0), ("array-645364531", 2, 2, 2), ("array-645364531", 2, 1, 1), ("array-645364531", 3, 1, 1), ("array-645364531", 3, 1, 2)]
    buffer_pos_list = list()
    for i, s in enumerate(slices_list):
        pos_in_buffer, slices = convert_proxy_to_buffer_slices(s, 'buffer-465316453-10-23', (slice(None, None, None), slice(None, None, None), slice(None, None, None)), array_to_original, original_array_chunks, original_array_blocks_shape)
        buffer_pos_list.append(pos_in_buffer)

    expected_list = [[[('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 1), ("buffer-465316453-10-23-proxy", buffer_pos_list[0][0], buffer_pos_list[0][1], buffer_pos_list[0][2])],
                    [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 2), ("buffer-465316453-10-23-proxy", buffer_pos_list[1][0], buffer_pos_list[1][1], buffer_pos_list[1][2])],
                    [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 3), ("buffer-465316453-10-23-proxy", buffer_pos_list[2][0], buffer_pos_list[2][1], buffer_pos_list[2][2])]],
                    [[('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 4), ("buffer-465316453-10-23-proxy", buffer_pos_list[3][0], buffer_pos_list[3][1], buffer_pos_list[3][2])],
                    [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 5), ("buffer-465316453-10-23-proxy", buffer_pos_list[4][0], buffer_pos_list[4][1], buffer_pos_list[4][2])],
                    [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 6), ("buffer-465316453-10-23-proxy", buffer_pos_list[5][0], buffer_pos_list[5][1], buffer_pos_list[5][2])]]]
    
    if _list != expected_list:
        print("error in", sys._getframe().f_code.co_name)
        print(_list, "\n\nexpected\n\n", expected_list)
        return
    
    print("success")


def test_update_io_tasks_getitem():
    buffer_node_name = 'buffer-465316453-0-6'
    array_to_original = {'array-6f870a321e8529128cb9bb82b8573db5': "array-original-645364531"}
    original_array_chunks = {"array-original-645364531": (10, 20, 30)}
    original_array_blocks_shape = {"array-original-645364531": (5, 3, 2)}

    getitem_graph = get_getitem_dict_from_proxy_array_sample()
    dependent_tasks = [('getitem-c6555b775be6a9d771866321a0d38252', 0, 0, 0),
                       ('getitem-c6555b775be6a9d771866321a0d38252', 0, 0, 1),
                       ('getitem-c6555b775be6a9d771866321a0d38252', 0, 0, 2),
                       ('getitem-c6555b775be6a9d771866321a0d38252', 0, 0, 3),
                       ('getitem-c6555b775be6a9d771866321a0d38252', 0, 0, 4),
                       ('getitem-c6555b775be6a9d771866321a0d38252', 0, 0, 5)]
    
    load = [(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4),(0,0,5)]
    shape = original_array_blocks_shape["array-original-645364531"]
    load = [_3d_to_numeric_pos(l, shape, order='C') for l in load]
    load = sorted(load)

    getitem_graph = update_io_tasks_getitem(getitem_graph, load, buffer_node_name, dependent_tasks, array_to_original, original_array_chunks, original_array_blocks_shape)

    dims = (10, 20, 30)
    expected_slices = [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (0,2,0), (0,2,1)]
    slices_in_buffer = (slice(None, None, None), slice(None, None, None), slice(None, None, None))
    

    for i, s in enumerate(expected_slices):
        pos_in_buffer, slices = convert_proxy_to_buffer_slices(tuple(['array-6f870a321e8529128cb9bb82b8573db5'] + list(s)), 
                                                                                buffer_node_name, 
                                                                                slices_in_buffer, 
                                                                                array_to_original, 
                                                                                original_array_chunks, 
                                                                                original_array_blocks_shape)
        expected_slices[i] = slices

    error = False
    for i, key in enumerate(list(getitem_graph.keys())):
        val = getitem_graph[key]
        f, target_key, slices = val
        target_name = target_key[0]
        target_index = target_key[1:]

        if target_name != buffer_node_name:
            print(target_name, buffer_node_name)
            error = True
            
        if tuple(target_index) != (0,0,0):
            print(target_index, (0,0,0))
            error = True

        if expected_slices[i] != slices:
            print(expected_slices, slices)
            error = True

    if error:
        print("error in", sys._getframe().f_code.co_name)
        return
    print("success")


def test_update_io_tasks_rechunk():
    buffer_node_name = 'buffer-465316453-0-30'
    array_to_original = {'array-3ec4eddf5e385f67eb8007734372b503': "array-original-645364531"}
    original_array_chunks = {"array-original-645364531": (10, 20, 30)}
    original_array_blocks_shape = {"array-original-645364531": (5, 3, 2)}
    rechunk_graph = get_rechunk_dict_from_proxy_array_sample(array_name='array-3ec4eddf5e385f67eb8007734372b503', array_names=None, add_list=[1,2,3])
    dependent_tasks = [('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 3),
                        ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 7),
                        ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 8),
                        ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 9),
                        ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 10),
                        ('rechunk-merge-a168f56ba79513b9ed87b2f22dd07458', 0, 0, 0)]

    load = [(0,0,0), (0,0,3), (0,1,3), (0,2,0), (0,2,1), (0,2,2), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,1,2)]
    shape = original_array_blocks_shape["array-original-645364531"]
    load = [_3d_to_numeric_pos(l, shape, order='C') for l in load]
    load = sorted(load)

    #prepare expected results for merge
    rechunk_targets = [(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2)]
    slices_in_buffer = (slice(None, None, None), slice(None, None, None), slice(None, None, None))
    for i, s in enumerate(rechunk_targets):
        pos_in_buffer, slices = convert_proxy_to_buffer_slices(tuple(['array-3ec4eddf5e385f67eb8007734372b503'] + list(s)), 
                                                                                buffer_node_name, 
                                                                                slices_in_buffer, 
                                                                                array_to_original, 
                                                                                original_array_chunks, 
                                                                                original_array_blocks_shape)
        rechunk_targets[i] = pos_in_buffer

    # compute result
    rechunk_key = 'rechunk-merge-a168f56ba79513b9ed87b2f22dd07458'
    graph = dict()
    graph[rechunk_key] = rechunk_graph
    graph = update_io_tasks_rechunk(graph, rechunk_key, 
                                            load,
                                            dependent_tasks,
                                            buffer_node_name,
                                            array_to_original,
                                            original_array_chunks, 
                                            original_array_blocks_shape)

    # compare to expected
    rechunk_targets_found = list()
    for k, v in rechunk_graph.items():
        if "rechunk-merge" in k[0]:
            concat_lists = v[1][0]
            for _list in concat_lists:
                for target_key in _list:
                    if "proxy" in target_key[0]:
                        proxy_index = tuple(target_key[1:])
                        rechunk_targets_found.append(proxy_index)
        elif "rechunk-split" in k[0]:
            f, target_key, slices = v
            if target_key[0] != buffer_node_name or target_key[1:] != tuple([0, 0, 0]):  # we don't check slices again because it is supposed to have been tested earlier
                print("error in", sys._getframe().f_code.co_name)
                print(target_key[0])
                print(target_key[1:])
                return

    if rechunk_targets != rechunk_targets_found:
        print("error in", sys._getframe().f_code.co_name)
        print(rechunk_targets)
        print(rechunk_targets_found)
        return

    print("success")


def test_update_io_tasks():
    buffer_node_name = 'buffer-465316453-0-30'
    proxy_array_name = 'array-6f870a321e8529128cb9bb82b8573db5'
    original_array_name = "array-original-645364531"
    array_to_original = {proxy_array_name: original_array_name}
    original_array_chunks = {original_array_name: (10, 20, 30)}
    original_array_blocks_shape = {original_array_name: (5, 3, 2)}
    rechunk_deps = [('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 3),
                        ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 7),
                        ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 8),
                        ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 9),
                        ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 10),
                        ('rechunk-merge-a168f56ba79513b9ed87b2f22dd07458', 0, 0, 0)]
    getitem_deps = [('getitem-c6555b775be6a9d771866321a0d38252', 0, 0, 0),
                       ('getitem-c6555b775be6a9d771866321a0d38252', 0, 0, 1),
                       ('getitem-c6555b775be6a9d771866321a0d38252', 0, 0, 2),
                       ('getitem-c6555b775be6a9d771866321a0d38252', 0, 0, 3),
                       ('getitem-c6555b775be6a9d771866321a0d38252', 0, 0, 4),
                       ('getitem-c6555b775be6a9d771866321a0d38252', 0, 0, 5)]

    deps_dict = {'array-6f870a321e8529128cb9bb82b8573db5': rechunk_deps + getitem_deps}
    getitem_d = get_getitem_dict_from_proxy_array_sample()
    rechunk_d = get_rechunk_dict_from_proxy_array_sample(array_name='array-6f870a321e8529128cb9bb82b8573db5', array_names=None, add_list=[1,2,3])

    graph = {
        'getitem-c6555b775be6a9d771866321a0d38252': getitem_d,
        'rechunk-merge-a168f56ba79513b9ed87b2f22dd07458': rechunk_d
    }

    load = [(0,0,0), (0,0,3), (0,1,3), (0,2,0), (0,2,1), (0,2,2), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,1,2)] + [(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4),(0,0,5)]
    shape = original_array_blocks_shape["array-original-645364531"]
    load = [_3d_to_numeric_pos(l, shape, order='C') for l in load]
    load = sorted(list(set(load)))

    #try:
    graph = update_io_tasks(graph, 
                        load, 
                        deps_dict, 
                        proxy_array_name, 
                        array_to_original, 
                        original_array_chunks, 
                        original_array_blocks_shape, 
                        buffer_node_name)

    """except:
        print("error in", sys._getframe().f_code.co_name)
        return """

    # printer
    """for k, v in graph.items():
        print("\n", k)
        for k_, v_ in v.items():
            print("\n\t", k_)
            print("\n\t", v_)"""
        
    print("success")


def test_create_buffer_node():

    proxy_array_name = 'array-6f870a321e8529128cb9bb82b8573db5'
    original_array_name = "array-original-645364531"
    array_to_original = {proxy_array_name: original_array_name}
    original_array_chunks = {original_array_name: (10, 20, 30)}
    original_array_blocks_shape = {original_array_name: (5, 3, 2)}

    dask_graph = dict()
    load = [4,5,6,7,8]

    try:
        dask_graph, merged_array_proxy_name = create_buffer_node(dask_graph, 
                                                                proxy_array_name, 
                                                                load, 
                                                                array_to_original,
                                                                original_array_blocks_shape, 
                                                                original_array_chunks)
    except:
        print("error in", sys._getframe().f_code.co_name)
        return

    for k, v in dask_graph.items():
        # print("\n", k)
        for k_, v_ in v.items():
            # print("\n\t", k_)
            # print("\n\t", v_)
            if v[('merged-part-4-8', 0, 0, 0)][2] != (slice(0, 20, None), slice(0, 60, None), slice(0, 60, None)):
                print("error in", sys._getframe().f_code.co_name)
                return

    print("success")


def test_create_buffers():
    proxy_array_name = 'array-6f870a321e8529128cb9bb82b8573db5'
    original_array_name = "array-original-645364531"
    array_to_original = {proxy_array_name: original_array_name}
    original_array_chunks = {original_array_name: (200, 300, 230)}
    original_array_blocks_shape = {original_array_name: (7, 5, 7)}
    slices_list = [0,1,2,3,4,5,12,13,14,17,20,21,22,23]

    """
    row size = 7 blocks
    slices size = 35 blocks

    shape = (200,300,230)
    row byte size = (200*300*230) * 7 blocks * 4 bytes= 13800000 * 7 * 4 = 386 400 000
    slice byte size = 386 400 000 * 5 = 1 932 000 000
    default mem = 1 000 000 000

    => strategy: buffer=row_size max => 7 blocks contiguous max
    rows: 0-6 7-13 14-20 21-27
    """
    expected = [[0,1,2,3,4,5], [12,13], [14], [17], [20], [21,22,23]]

    buffers = create_buffers(slices_list, proxy_array_name, array_to_original, original_array_chunks, original_array_blocks_shape, nb_bytes_per_val=8)
    if buffers != expected:
        print("error in", sys._getframe().f_code.co_name)
        print("buffer'\n", buffers, "\n\n")
        print("expected\n", expected)
        return 

    slices_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    buffers = create_buffers(slices_list, proxy_array_name, array_to_original, original_array_chunks, original_array_blocks_shape, nb_bytes_per_val=8)
    expected = [[0,1,2,3,4,5,6], [7,8,9,10,11,12,13], [14,15,16,17,18,19,20], [21,22,23,24]]
    if buffers != expected:
        print("error in", sys._getframe().f_code.co_name)
        print("buffer'\n", buffers, "\n\n")
        print("expected\n", expected)
        return 

    print("success")

def test_is_in_load():
    proxy_array_name = 'array-6f870a321e8529128cb9bb82b8573db5'
    original_array_name = "array-original-645364531"
    array_to_original = {proxy_array_name: original_array_name}
    original_array_chunks = {original_array_name: (10, 20, 30)}
    original_array_blocks_shape = {original_array_name: (7, 5, 7)}
    load = range(6,36)

    proxy_key = (proxy_array_name, 0, 2, 2)
    result = is_in_load(proxy_key, load, array_to_original, original_array_blocks_shape)
    if result != True:
        print("error in", sys._getframe().f_code.co_name)
        return 

    proxy_key = (proxy_array_name, 1, 0, 1)
    result = is_in_load(proxy_key, load, array_to_original, original_array_blocks_shape)
    if result != False:
        print("error in", sys._getframe().f_code.co_name)
        return 
    
    print("success")