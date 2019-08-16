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
        for e in l:
            try: # iterable 
                iterator = iter(e)
                plain_list = decompose_iterable(e, plain_list)
            except TypeError: # not iterable
                pass
        return plain_list

    def is_task(v):
        if isinstance(val, tuple) and callable(val[0]):
            return True 
        return False

    def add_to_remade_graph(remade_graph, key, value, undir):
        """
        arg: 
            undir: do you want undirected graph
        """
        add_to_dict_of_lists(remade_graph, key, value, unique=True)
        if undir:
            add_to_dict_of_lists(remade_graph, value, key, unique=True)

    for k, v in graph.items():  
        # if it is a subgraph, recurse
        if isinstance(v, dict):
            remade_graph = get_graph_from_dask(graph, undirected=undirected)

        # if it is a task, add its arguments
        elif is_task(v):  
            for arg in v[1:]:
                try: # iterable 
                    iterator = iter(arg)
                    l = decompose_iterable(arg, list())
                    for e in l:
                        add_to_remade_graph(remade_graph, key, e, undirected)
                except TypeError: # not iterable
                    add_to_remade_graph(remade_graph, key, arg, undirected)                         

        # if it is an argument, add it
        else:  
            add_to_remade_graph(remade_graph, key, val, undirected)

    return remade_graph


def get_used_proxies(graph, undirected):
    """ go through graph and find the proxies that are used by other tasks
    proxy: task that getitem directly from original-array
    """

    remade_graph = get_graph_from_dask(dict(), undirected=undirected)

    # find proxies
    global proxies_keys = list()
    for k, v in remade_graph.items():
        try:
            f, target, slices = v
            if "array-original" in target and all([isinstance(s, slice) for s in slices]):
                proxies_keys.append(k)
        except:
            pass

    # find used proxies
    used_proxies = list()
    for k, v in remade_graph.items():
        if v in proxies_keys:
            used_proxies.append(v)

    """"
    def is_target(x):
        if x in proxies_keys:
            return True
        return False
    
    connected_comps = BFS_connected_components(remade_graph, 
                                               filter_condition_for_root_nodes=is_target)
    max_len = max(map(len, connected_comps.values()))
    main_components = [
        _list for comp, _list in connected_comps.items() if len(_list) == max_len]

    # get used proxies from main components
    used_proxies = list()
    for main_comp in main_components:
        for e in main_comp:
            if is_target(e):
                used_proxies.append(e)
    """"

    # create dictionaries
    origarr_to_slices_dict = dict()
    origarr_to_used_proxies_dict = dict()
    for k in used_proxies:
        v = remade_graph[k]
        f, target, slices = v
        add_to_dict_of_lists(origarr_to_slices_dict, target, slices, unique=True)
        add_to_dict_of_lists(origarr_to_used_proxies_dict, target, k, unique=True)

    return (origarr_to_slices_dict, 
            origarr_to_used_proxies_dict)