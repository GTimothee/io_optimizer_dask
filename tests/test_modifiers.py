import os

import tests_utils
from tests_utils import get_test_arr, CaseConfig, ONE_GIG, neat_print_graph

import optimize_io
from optimize_io.modifiers import *


def test_add_to_dict_of_lists():
    d = {'a': [1], 'c': [5, 6]}
    d = add_to_dict_of_lists(d, 'b', 2)
    d = add_to_dict_of_lists(d, 'b', 3)
    d = add_to_dict_of_lists(d, 'c', 4)
    expected = {'a': [1], 'b': [2, 3], 'c': [5, 6, 4]}
    assert expected == d


def test_get_array_block_dims():
    shape = (500, 1200, 300)
    chunks = (100, 300, 20)
    block_dims = get_array_block_dims(shape, chunks)
    expected = (5, 4, 15)
    assert block_dims == expected


def test_decompose_iterable():
    l = [0, [1, 2, 3], 4, [5], [[6, 7]]]
    assert flatten_iterable(l) == list(range(8))


def test_get_graph_from_dask():
    # create config for the test
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
    new_config = CaseConfig(opti=None, 
                             scheduler_opti=None, 
                             out_path=None, 
                             buffer_size=ONE_GIG, 
                             input_file_path=data, 
                             chunk_shape=None)
    new_config.sum_case(nb_chunks=2)
    dask_array = get_test_arr(new_config)

    # test function
    dask_graph = dask_array.dask.dicts 
    graph = get_graph_from_dask(dask_graph, undirected=False)

    with open('tests/outputs/remade_graph.txt', "w+") as f:
        for k, v in graph.items():
            f.write("\n\n" + str(k))
            f.write("\n" + str(v))

    for k, v in graph.items():
        print("\nkey", k)
        print("value", v)


def test_get_used_proxies():
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
    new_config = CaseConfig(opti=None, 
                             scheduler_opti=None, 
                             out_path=None, 
                             buffer_size=ONE_GIG, 
                             input_file_path=data, 
                             chunk_shape=None)
    new_config.sum_case(nb_chunks=2)
    

    for use_BFS in [True, False]:
        dask_array = get_test_arr(new_config)

        # test function
        dask_graph = dask_array.dask.dicts 
        dicts = get_used_proxies(dask_graph, use_BFS=True)
        
        # test slices values
        slices = list(dicts['proxy_to_slices'].values())
        s1 = (slice(0, 220, None), slice(0, 242, None), slice(0, 200, None))
        s2 = (slice(0, 220, None), slice(0, 242, None), slice(200, 400, None))

        #print(dicts['origarr_to_used_proxies'])

        print("slices", slices)

        assert slices == [s1, s2]

        # test proxies indices
        proxy_indices = list()
        for l in dicts['origarr_to_used_proxies'].values():
            for t in l:
                proxy_indices.append(tuple(t[1:]))
        assert set(proxy_indices) == set([(0, 0, 0), (0, 0, 1)])


# not finished
"""def get_used_proxies_rechunk_case():
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
    new_config = CaseConfig(opti=None, 
                             scheduler_opti=None, 
                             out_path=None, 
                             buffer_size=ONE_GIG, 
                             input_file_path=data, 
                             chunk_shape=(770, 605, 700))
    new_config.sum_case(nb_chunks=2)
    dask_array = get_test_arr(new_config)

    # test function
    dask_graph = dask_array.dask.dicts 
    dicts = get_used_proxies(dask_graph, undirected=False, use_BFS=True)
    slices = list(dicts['proxy_to_slices'].values())
    with open('tests/outputs/slices_found.txt', "w+") as f:
        for s in slices:
            f.write(str(s) + "\n")"""


"""def test_BFS_connected_comp():
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
    new_config = CaseConfig(opti=None, 
                             scheduler_opti=None, 
                             out_path=None, 
                             buffer_size=ONE_GIG, 
                             input_file_path=data, 
                             chunk_shape=(770, 605, 700)) # (660, 726, 600))
    new_config.sum_case(nb_chunks=2)
    dask_array = get_test_arr(new_config)
    dask_array.visualize(filename='tests/outputs/img.png', optimize_graph=False)

    dask_graph = dask_array.dask.dicts 

    # get remade graph 
    graph = get_graph_from_dask(dask_graph, undirected=True)
    with open('tests/outputs/remade_graph.txt', "w+") as f:
        for k, v in graph.items():
            f.write("\n\n" + str(k))
            f.write("\n" + str(v))

    # get connected components with BFS
    connected_comps = BFS_connected_components(graph,
                        filter_condition_for_root_nodes=true_dumb_function,
                        max_iterations=10)

    # get main components
    max_len = max(map(len, connected_comps.values()))
    main_components = [
        _list for comp,
        _list in connected_comps.items() if len(_list) == max_len]

    print("MAIN components")
    for m in main_components:
        print("\nlen(m)", len(m))
        nb_proxies =0
        for e in m:
            if isinstance(e, tuple) and isinstance(e[0], str) and "array" in e[0]:
                print(e)
                nb_proxies +=1
        print("found", nb_proxies, "proxies")"""


def test_BFS():
    graph = {
        'a': ['b', 'c'],
        'b': [],
        'c': ['d', 'e'],
        'd': [],
        'e': []
    }
    values = standard_BFS('a', graph)
    assert values == ['a', 'b', 'c', 'd', 'e']

