import collections

import optimize_io
from optimize_io.get_slices import *
from optimize_io.get_dicts import *

import numpy as np

def decompose_iterable(l, plain_list=list()):
    """ transform iterables and multistage iterables into one list
    ex of iterable: list
    ex of multistage iterable: list of lists
    ex: 
        >> x = decompose_iterable([a, [b, [c]]], list()) 
        >> print(x)
        [a, b, c]
    """
    for e in l:
        print(l)
        if not isinstance(e, str) and not isinstance(e, np.ndarray):
            print(type(e))
            try: # iterable 
                iterator = iter(e)
                plain_list = decompose_iterable(e, plain_list)
            except TypeError: # not iterable
                plain_list.append(e)
    return plain_list


def get_graph_from_dask(graph, undirected=False):
    """ Transform dask graph into a real graph in order to use graph algorithms on it.
    in the graph, each node is an element (tuple, list, object)
    when directed, each edge is the relation "is the result of a function on"
    {a: b} implies that a is the result of a function a = f(b or more)
    each key is mapped to a list of values (other keys used by this key as input)
    """

    def is_task(v):
        if isinstance(v, tuple) and callable(v[0]):
            return True 
        return False

    def add_to_remade_graph(d, key, value, undir):
        """
        arg: 
            undir: do you want undirected graph
        """
        d = add_to_dict_of_lists(d, key, value, unique=True)
        if undir:
            d = add_to_dict_of_lists(d, value, key, unique=True)

    remade_graph = dict()
    for key, v in graph.items():  
        # if it is a subgraph, recurse
        if isinstance(v, dict):
            subgraph = get_graph_from_dask(v, undirected=undirected)
            remade_graph.update(subgraph)

        # if it is a task, add its arguments
        elif is_task(v):  
            for arg in v[1:]:
                if not isinstance(arg, str) and not isinstance(arg, tuple) and not isinstance(arg, int):
                    try: # iterable 
                        iterator = iter(arg)
                        l = decompose_iterable(arg)
                        for e in l:
                            add_to_remade_graph(remade_graph, key, e, undirected)
                        continue
                    except TypeError: # not iterable
                        pass                        
                add_to_remade_graph(remade_graph, key, arg, undirected)
        # if it is an argument, add it
        elif isinstance(key, collections.Hashable) and isinstance(v, collections.Hashable):  
            add_to_remade_graph(remade_graph, key, v, undirected)
        else:
            pass

    return remade_graph


def get_used_proxies(graph, undirected):
    """ go through graph and find the proxies that are used by other tasks
    proxy: task that getitem directly from original-array
    """

    def get_unused_keys(remade_graph):
        """ find keys in the graph that are not used as values by another(other) key(s)
        """
        keys = list(remade_graph.keys())
        vals = list(remade_graph.values())
        flatten = list()

        # flatten the values which is a list of lists 
        # because get_graph_from_dask which is using add_to_dict_of_lists
        for l in vals:
            for e in l:
                flatten.append(e)

        # do the actual job
        unused_keys = list()
        for key in keys:
            if key not in flatten:
                unused_keys.append(key)

        return unused_keys

    remade_graph = get_graph_from_dask(graph, undirected=undirected)
    unused_keys = get_unused_keys(remade_graph)

    proxy_to_slices = dict()
    origarr_to_used_proxies = dict()
    origarr_to_obj = dict()
    origarr_to_blocks_shape = dict()
    proxy_to_dict = dict()
    for k, v in remade_graph.items():
        if k not in unused_keys:
            # if it is an array_original, add its data to dictionaries
            if isinstance(k, str) and "array-original" in k:
                obj = v.pop(0)
                origarr_to_obj[k] = obj
                origarr_to_blocks_shape[k] = get_array_block_dims(obj.shape, obj.chunks)
                continue

            try:
                target, slices = v
                # search for values that are array-original, meaning that key is proxy 
                if "array-original" in target and all([isinstance(s, slice) for s in slices]):
                    v = remade_graph[k]
                    target, slices = v

                    add_to_dict_of_lists(proxy_to_slices, k, slices, unique=True)
                    add_to_dict_of_lists(origarr_to_used_proxies, target, k, unique=True)
                    add_to_dict_of_lists(proxy_to_dict, k, remade_graph, unique=True)
                    continue
            except:
                pass

    return {
        'proxy_to_slices': proxy_to_slices, 
        'origarr_to_used_proxies': origarr_to_used_proxies,
        'origarr_to_obj': origarr_to_obj,
        'origarr_to_blocks_shape': origarr_to_blocks_shape,
        'proxy_to_dict': proxy_to_dict
    }