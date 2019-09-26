import os
import dask
import collections
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

def get_config_chunk_shape():
    try:
        optimization = dask.config.get("io-optimizer")
        return dask.config.get("io-optimizer.chunk_shape")
    except BaseException:
        return (220, 242, 200)
        

def get_array_block_dims(shape):
    """ from shape of image and size of chukns=blocks, return the dimensions of the array in terms of blocks
    i.e. number of blocks in each dimension
    """
    chunks = get_config_chunk_shape()
    if not len(shape) == len(chunks):
        raise ValueError(
            "chunks and shape should have the same dimension",
            shape,
            chunks)
    return tuple([int(s / c) for s, c in zip(shape, chunks)])


def flatten_iterable(l, plain_list=list()):
    for e in l:
        if isinstance(e, list) and not isinstance(e, (str, bytes)):
            plain_list = flatten_iterable(e, plain_list)
        else:
            plain_list.append(e)
    return plain_list


def standard_BFS(root, graph):
    nodes = list(graph.keys())
    queue = [(root, 0)]
    visited = [root]

    with open('tests/outputs/BFS_out.txt', 'w+') as f:
        max_depth = 0
        while len(queue) > 0:
            node, depth = queue.pop(0)
            f.write("\n\n\n-------------- current node" + str(node))

            # not related
            if depth > max_depth:
                max_depth = depth

            # specific to our problem (because of remade graph)
            if not isinstance(node, collections.Hashable) or not node in graph:  
                continue

            # algorithm
            neighbors = graph[node]
            for n in neighbors:
                f.write("\n\nvisited:")
                if not n in visited:
                    queue.append((n, depth + 1))
                    visited.append(n)
                    f.write("\n" + str(n))
            """f.write("\nqueue:" + str(queue))
            f.write("\nlen queue:" + str(len(queue)))"""
    
    return visited, max_depth


def is_task(v):
    if isinstance(v, tuple) and callable(v[0]):
        return True 
    return False


def get_graph_from_dask(graph, undirected=False):
    """ Transform dask graph into a real graph in order to use graph algorithms on it.
    in the graph, each node is an element (tuple, list, object)
    when directed, each edge is the relation "is the result of a function on"
    {a: b} implies that a is the result of a function a = f(b or more)
    each key is mapped to a list of values (other keys used by this key as input)
    """

    def add_to_remade_graph(d, key, value, undir):
        """
        arg: 
            undir: do you want undirected graph
        """
        try:  # do not treat slices
            if (isinstance(key, tuple) and all([isinstance(s, slice) for s in key])) or (isinstance(value, tuple) and all([isinstance(s, slice) for s in value])):
                return 
        except:
            pass

        d = add_to_dict_of_lists(d, key, value, unique=True)
        if undir:
            d = add_to_dict_of_lists(d, value, key, unique=True)

    remade_graph = dict()
    for key, v in graph.items():  
        # if it is a subgraph, recurse
        if isinstance(v, dict):
            if isinstance(key, str) and "array-original" in key:
                add_to_remade_graph(remade_graph, key, v, undirected)
            else:
                subgraph = get_graph_from_dask(v, undirected=undirected)
                remade_graph.update(subgraph)

        # if it is a task, add its arguments
        elif is_task(v):  
            for arg in v[1:]:
                if isinstance(arg, tuple):
                    pass

                if isinstance(arg, list):
                    l = flatten_iterable(arg)
                    for e in l:
                        if isinstance(e, tuple):
                            pass 
                        if isinstance(key, collections.Hashable) and isinstance(e, collections.Hashable):                 
                            add_to_remade_graph(remade_graph, key, e, undirected)
                    continue

                if isinstance(key, collections.Hashable) and isinstance(arg, collections.Hashable):   
                    add_to_remade_graph(remade_graph, key, arg, undirected)

        # if it is an argument, add it
        elif isinstance(key, collections.Hashable):
            if isinstance(v, collections.Hashable):  
                if isinstance(v, tuple):
                    pass
                add_to_remade_graph(remade_graph, key, v, undirected)
        else:
            pass
    
    return remade_graph


#TODO : refactor
def search_dask_graph(graph, proxy_to_slices, proxy_to_dict, origarr_to_used_proxies, origarr_to_obj, origarr_to_blocks_shape, unused_keys, main_components=None):
    """ Search proxies in the remade graph and fill in dictionaries to store information.
    """

    for key, v in graph.items():  

        # if it is a subgraph, recurse
        if isinstance(v, dict):
            search_dask_graph(v, proxy_to_slices, proxy_to_dict, origarr_to_used_proxies, origarr_to_obj, origarr_to_blocks_shape, unused_keys, main_components)

        # if it is an original array, store it
        elif isinstance(key, str) and "array-original" in key: # TODO: support other formats
            obj = v
            origarr_to_obj[key] = obj
            if not obj.shape:
                raise ValueError("Empty dataset!")
            origarr_to_blocks_shape[key] = get_array_block_dims(obj.shape)
            continue

        # if it is a task, add its arguments
        elif is_task(v) and (key not in unused_keys):  
            if main_components:
                used_key = False
                for main_comp in main_components:
                    if key in main_comp:
                        used_key = True 
            else:
                used_key = True

            if used_key:
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


def get_used_proxies(graph, use_BFS=True):
    """ go through graph and find the proxies that are used by other tasks
    proxy: task that getitem directly from original-array
    """
    if not use_BFS:
        remade_graph = get_graph_from_dask(graph, undirected=False)
        unused_keys = get_unused_keys(remade_graph)
        main_components = None
    else:
        remade_graph = get_graph_from_dask(graph, undirected=False)
        root_nodes = get_unused_keys(remade_graph)
        main_components = list()
        max_depth = 0
        for root in root_nodes:
            node_list, depth = standard_BFS(root, remade_graph)
            if len(main_components) == 0 or depth > max_depth:
                main_components = [node_list]
                max_depth = depth
            elif depth == max_depth:
                main_components.append(node_list)
        unused_keys = list()

    proxy_to_slices = dict()
    origarr_to_used_proxies = dict()
    origarr_to_obj = dict()
    origarr_to_blocks_shape = dict()
    proxy_to_dict = dict()
    search_dask_graph(graph, proxy_to_slices, proxy_to_dict, origarr_to_used_proxies, origarr_to_obj, origarr_to_blocks_shape, unused_keys, main_components)

    return {
        'proxy_to_slices': proxy_to_slices, 
        'origarr_to_used_proxies': origarr_to_used_proxies,
        'origarr_to_obj': origarr_to_obj,
        'origarr_to_blocks_shape': origarr_to_blocks_shape,
        'proxy_to_dict': proxy_to_dict
    }