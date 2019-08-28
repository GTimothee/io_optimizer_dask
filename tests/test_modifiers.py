import tests_utils
from tests_utils import get_test_arr

import optimize_io
from optimize_io.modifiers import add_to_dict_of_lists, get_array_block_dims, flatten_iterable, get_graph_from_dask, get_used_proxies


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
    dask_array = get_test_arr(case='sum', nb_arr=2)

    # test function
    dask_graph = dask_array.dask.dicts 
    graph = get_graph_from_dask(dask_graph, undirected=False)

    for k, v in graph.items():
        print("\nkey", k)
        print("value", v)


def test_get_used_proxies():
    dask_array = get_test_arr(case='sum', nb_arr=2)

    # test function
    dask_graph = dask_array.dask.dicts 
    dicts = get_used_proxies(dask_graph, undirected=False)
    
    slices = list()
    for s in dicts['proxy_to_slices'].values():
        slices.append(s)
    s1 = (slice(0, 220, None), slice(0, 242, None), slice(0, 200, None))
    s2 = (slice(0, 220, None), slice(0, 242, None), slice(200, 400, None))

    print(slices)

    assert slices == [s1, s2]

    proxy_indices = list()
    for l in dicts['origarr_to_used_proxies'].values():
        for t in l:
            proxy_indices.append(tuple(t[1:]))
    assert set(proxy_indices) == set([(0, 0, 0), (0, 0, 1)])