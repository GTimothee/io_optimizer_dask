import os

import tests_utils
from tests_utils import get_test_arr, CaseConfig, ONE_GIG, neat_print_graph, SUB_BIGBRAIN_SHAPE

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
    dask.config.set({
        'io-optimizer': {
            'chunk_shape': chunks
        }
    })
    block_dims = get_array_block_dims(shape)
    expected = (5, 4, 15)
    assert block_dims == expected


def test_decompose_iterable():
    l = [0, [1, 2, 3], 4, [5], [[6, 7]]]
    assert flatten_iterable(l, list()) == list(range(8))


def test_get_graph_from_dask():
    # create config for the test
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
    new_config = CaseConfig(opti=None, 
                             scheduler_opti=None, 
                             out_path=None, 
                             buffer_size=ONE_GIG, 
                             input_file_path=data, 
                             chunk_shape=None)
    new_config.create_or_overwrite(None, SUB_BIGBRAIN_SHAPE, overwrite=False)
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
    new_config.create_or_overwrite(None, SUB_BIGBRAIN_SHAPE, overwrite=False)
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


def test_BFS():
    graph = {
        'a': ['b', 'c'],
        'b': [],
        'c': ['d', 'e'],
        'd': [],
        'e': [],
        'f': ['e']
    }
    values, depth = standard_BFS('a', graph)
    assert values == ['a', 'b', 'c', 'd', 'e']
    assert depth == 2

    values, depth = standard_BFS('f', graph)
    assert values == ['f', 'e']
    assert depth == 1


def test_get_unused_keys():
    graph = {
        'a': ['b', 'c'],
        'b': [],
        'c': ['d', 'e'],
        'd': [],
        'e': [],
        'f': ['e']
    }
    root_nodes = get_unused_keys(graph)
    assert root_nodes == ['a', 'f']


def test_BFS_2():
    """ test to include bfs in the program
    """
    graph = {
        'a': ['b', 'c'],
        'b': [],
        'c': ['d', 'e'],
        'd': [],
        'e': [],
        'f': ['e']
    }
    root_nodes = get_unused_keys(graph)

    max_components = list()
    max_depth = 0
    for root in root_nodes:
        node_list, depth = standard_BFS(root, graph)
        if len(max_components) == 0 or depth > max_depth:
            max_components = [node_list]
            max_depth = depth
        elif depth == max_depth:
            max_components.append(node_list)

    assert max_depth == 2
    assert len(max_components) == 1
    assert max_components[0] == ['a', 'b', 'c', 'd', 'e']


def test_BFS_3():
    """ test to include bfs in the program and test with rechunk case
    """

    # get test array with rechunking
    data = os.path.join(os.getenv('DATA_PATH'), 'sample_array.hdf5')
    new_config = CaseConfig(opti=None, 
                             scheduler_opti=None, 
                             out_path=None, 
                             buffer_size=ONE_GIG, 
                             input_file_path=data, 
                             chunk_shape=(770, 605, 700))  # (660, 726, 600))
    new_config.create_or_overwrite(None, SUB_BIGBRAIN_SHAPE, overwrite=False)
    new_config.sum_case(nb_chunks=2)
    dask_array = get_test_arr(new_config)
    dask_array.visualize(filename='tests/outputs/img.png', optimize_graph=False)

    # get formatted graph for processing
    dask_graph = dask_array.dask.dicts 
    graph = get_graph_from_dask(dask_graph, undirected=False)  # we want a directed graph

    with open('tests/outputs/remade_graph.txt', "w+") as f:
        for k, v in graph.items():
            f.write("\n\n" + str(k))
            f.write("\n" + str(v))

    # test the actual program
    root_nodes = get_unused_keys(graph)
    print('\nRoot nodes:')
    for root in root_nodes:
        print(root)

    max_components = list()
    max_depth = 0
    for root in root_nodes:
        node_list, depth = standard_BFS(root, graph)
        if len(max_components) == 0 or depth > max_depth:
            max_components = [node_list]
            max_depth = depth
        elif depth == max_depth:
            max_components.append(node_list)


    print("nb components found:", str(len(max_components)))

    #print("\nContent found:\n")
    nb_proxies = 0
    for node in max_components[0]:
        if isinstance(node, tuple) and isinstance(node[0], str) and 'array' in node[0]:
            #print(node) 
            nb_proxies += 1

    print("nb proxies found: " + str(nb_proxies))

    #TODO: assertions
    return


if __name__ == "__main__":
    test_get_used_proxies()