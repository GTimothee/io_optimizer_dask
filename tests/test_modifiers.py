import tests_utils
from tests_utils import *

import optimize_io
from optimize_io.modifiers import get_graph_from_dask, get_used_proxies

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
    r = get_used_proxies(dask_graph, undirected=False)
    origarr_to_slices_dict = r[0]
    origarr_to_used_proxies_dict = r[1]

    print("\norigarr_to_slices_dict", origarr_to_slices_dict)
    print("\n")
    print("origarr_to_used_proxies_dict", origarr_to_used_proxies_dict)

    expected_1 = {'array-original-id': [(slice(0, 220, None), slice(0, 242, None), slice(0, 200, None)), 
                                        (slice(0, 220, None), slice(0, 242, None), slice(200, 400, None))]}   

    assert len(list(origarr_to_slices_dict.keys())) == 1
    assert len(list(origarr_to_used_proxies_dict.keys())) == 1
    assert origarr_to_slices_dict[list(origarr_to_slices_dict.keys())[0]] == expected_1['array-original-id']

    proxy_indexes = list()
    for k, v in origarr_to_used_proxies_dict.items():
        assert 'array-original' in k
        for e in v:
                assert e[0].split('-')[0] == 'array' 
                proxy_indexes.append(tuple(e[1:]))
    assert proxy_indexes == [(0, 0, 0), (0, 0, 1)]