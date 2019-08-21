import tests_utils
from tests_utils import *

import optimize_io
from optimize_io.modifiers import flatten_iterable, get_graph_from_dask, get_used_proxies


def test_decompose_iterable():
    l = [0, [1, 2, 3], 4, [5], [[6, 7]]]
    assert flatten_iterable(l) == list(range(8))


def test_get_graph_from_dask():
    # load array
    data_path = get_test_array()
    key = 'data'
    arr = get_dask_array_from_hdf5(data_path, key)

    # create case
    nb_arr_to_sum = 2
    chunk_shape = 'blocks_dask_interpol'
    dask_array = add_chunks(arr, chunk_shape, number_of_arrays=nb_arr_to_sum)
    result_non_opti = dask_array.sum()

    # test function
    dask_graph = dask_array.dask.dicts 
    graph = get_graph_from_dask(dask_graph, undirected=False)

    for k, v in graph.items():
        print("\nkey", k)
        print("value", v)


def test_get_used_proxies():
    # load array
    data_path = get_test_array()
    key = 'data'
    arr = get_dask_array_from_hdf5(data_path, key)

    # create case
    nb_arr_to_sum = 2
    chunk_shape = 'blocks_dask_interpol'
    dask_array = add_chunks(arr, chunk_shape, number_of_arrays=nb_arr_to_sum)
    result_non_opti = dask_array.sum()

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