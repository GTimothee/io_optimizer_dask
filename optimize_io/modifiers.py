import collections

import optimize_io
from optimize_io.get_slices import *
from optimize_io.get_dicts import *


def get_graph_from_dask(graph, undirected=False):
    """ Transform dask graph into a real graph in order to use graph algorithms on it.
    in the graph, each node is an element (tuple, list, object)
    when directed, each edge is the relation "is the result of a function on"
    {a: b} implies that a is the result of a function a = f(b or more)
    """

    def decompose_iterable(l, plain_list):
        """ transform iterables and multistage iterables into one list
        ex of iterable: list
        ex of multistage iterable: list of lists
        ex: 
            >> x = decompose_iterable([a, [b, [c]]], list()) 
            >> print(x)
            [a, b, c]
        """
        print("list", l)
        for e in l:
            if not isinstance(e, str):
                try: # iterable 
                    iterator = iter(e)
                    plain_list = decompose_iterable(e, plain_list)
                except TypeError: # not iterable
                    plain_list.append(e)
        return plain_list

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
                        l = decompose_iterable(arg, list())
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

    remade_graph = get_graph_from_dask(graph, undirected=undirected)

    # find proxies
    global proxies_keys 
    proxies_keys = list()
    for k, v in remade_graph.items():
        try:
            target, slices = v
            if "array-original" in target and all([isinstance(s, slice) for s in slices]):
                proxies_keys.append(k)
        except:
            pass

    # find used proxies
    used_proxies = list()
    for k, v in remade_graph.items():
        if v[0] in proxies_keys:
            used_proxies.append(v[0])

    # create dictionaries
    origarr_to_slices_dict = dict()
    origarr_to_used_proxies_dict = dict()
    for k in used_proxies:
        v = remade_graph[k]
        target, slices = v
        add_to_dict_of_lists(origarr_to_slices_dict, target, slices, unique=True)
        add_to_dict_of_lists(origarr_to_used_proxies_dict, target, k, unique=True)

    return (origarr_to_slices_dict, 
            origarr_to_used_proxies_dict)