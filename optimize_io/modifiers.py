
import collections
from collections import Iterable
import numpy as np


def add_to_dict_of_lists(d, k, v, unique=False):
    """ if key does not exist, add a new list [value], else, 
    append value to existing list corresponding to the key
    """
    if k not in d:
        if v:
            d[k] = [v]
        else:
            d[k] = list()
    else:
        if v and (unique and v not in d[k]) or not unique:
            d[k].append(v)
    return d


def get_array_block_dims(shape, chunks):
    """ from shape of image and size of chukns=blocks, return the dimensions of the array in terms of blocks
    i.e. number of blocks in each dimension
    """
    if not len(shape) == len(chunks):
        raise ValueError(
            "chunks and shape should have the same dimension",
            shape,
            chunks)
    return tuple([int(s / c) for s, c in zip(shape, chunks)])


def flatten_iterable(l, plain_list=list()):
    for e in l:
        if isinstance(e, Iterable) and not isinstance(e, (str, bytes)):
            plain_list = flatten_iterable(e, plain_list)
        else:
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
                        l = flatten_iterable(arg)
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


def search_dask_graph(graph, proxy_to_slices, proxy_to_dict, origarr_to_used_proxies, origarr_to_obj, origarr_to_blocks_shape, unused_keys):
    """ 
    """

    def is_task(v):
        if isinstance(v, tuple) and callable(v[0]):
            return True 
        return False

    for key, v in graph.items():  
        # if it is a subgraph, recurse
        if isinstance(v, dict):
            search_dask_graph(v, proxy_to_slices, proxy_to_dict, origarr_to_used_proxies, origarr_to_obj, origarr_to_blocks_shape, unused_keys)

        elif isinstance(key, str) and "array-original" in key:
            obj = v
            origarr_to_obj[key] = obj
            origarr_to_blocks_shape[key] = get_array_block_dims(obj.shape, obj.chunks)
            continue

        # if it is a task, add its arguments
        elif is_task(v) and (key not in unused_keys):  
            try:
                f, target, slices = v

                # search for values that are array-original, meaning that key is proxy 
                if "array-original" in target and all([isinstance(s, slice) for s in slices]):
                    add_to_dict_of_lists(origarr_to_used_proxies, target, key, unique=True)
                    proxy_to_slices[key] = slices
                    proxy_to_dict[key] = graph
                    continue
            except:
                pass
        else:
            pass

    return 


def get_used_proxies(graph, undirected=False):
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
    search_dask_graph(graph, proxy_to_slices, proxy_to_dict, origarr_to_used_proxies, origarr_to_obj, origarr_to_blocks_shape, unused_keys)

    return {
        'proxy_to_slices': proxy_to_slices, 
        'origarr_to_used_proxies': origarr_to_used_proxies,
        'origarr_to_obj': origarr_to_obj,
        'origarr_to_blocks_shape': origarr_to_blocks_shape,
        'proxy_to_dict': proxy_to_dict
    }