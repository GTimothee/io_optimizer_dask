import os
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
        if isinstance(e, list) and not isinstance(e, (str, bytes)):
            plain_list = flatten_iterable(e, plain_list)
        else:
            plain_list.append(e)
    return plain_list


# to delete
def true_dumb_function(x):
    return True


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
            if not node in graph:  
                continue

            # algorithm
            neighbors = graph[node]
            for n in neighbors:
                f.write("\n\nvisited:")
                if not n in visited:
                    queue.append((n, depth + 1))
                    visited.append(n)
                    f.write("\n" + str(n))
                    f.write("\nqueue:" + str(queue))
            f.write("\nlen queue:" + str(len(queue)))
    
    return visited, max_depth


def BFS_connected_components(
        graph,
        filter_condition_for_root_nodes=true_dumb_function,
        max_iterations=10):
    """
    graph: undirected
    root node is at filter_condition_for_root_nodes 
    returns a dictionary with key = component id (increasing int) and value is a list of the nodes in the component
    """
    def all_nodes_not_visited(nb_nodes_visited, nb_nodes_total):
        if nb_nodes_visited != nb_nodes_total:
            return True
        return False

    def visit_node(visited, n, nb_nodes_visited, components, node_queue):
        visited[n] = True
        nb_nodes_visited += 1
        add_to_dict_of_lists(
            components, component_id, n, unique=True)
        node_queue.append(n)
        return nb_nodes_visited

    # initialize visitation dict
    nodes = list()
    for k in list(graph.keys()):
        if isinstance(k, str) and "array-original" in k:
            continue 
        nodes.append(k)

    nb_nodes_total = len(nodes)
    
    visited = dict(zip(nodes, nb_nodes_total * [False]))
    nb_nodes_visited = 0

    # initialize component variables
    components = dict()
    component_id = 0
    nb_its = 0
    if not os.path.isdir('tests/outputs'):
        os.makedirs('tests/outputs')
    f = open('tests/outputs/BFS_out.txt', 'w+')
    while all_nodes_not_visited(nb_nodes_visited, nb_nodes_total):

        # get next unvisited node (next start of connected component)
        node_queue = list()
        for n in nodes:
            if not visited[n]:
                if filter_condition_for_root_nodes(n):
                    f.write("\n\nroot node" + str(n))
                    nb_nodes_visited = visit_node(visited, n, nb_nodes_visited, components, node_queue)
                    break

        # run BFS
        while len(node_queue) > 0:
            curr_node = node_queue.pop(0)
            for n in graph[curr_node]:
                try:
                    if not visited[n]:
                        f.write("\nvisited" + str(n))
                        nb_nodes_visited = visit_node(visited, n, nb_nodes_visited, components, node_queue)
                except:
                    pass 

        component_id += 1

        # print("it", nb_its)
        """nb_its += 1
        if max_iterations == nb_its:
            break"""

    f.close()
    return components


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
            if not "array-original" in key:
                subgraph = get_graph_from_dask(v, undirected=undirected)
                remade_graph.update(subgraph)
            else:
                add_to_remade_graph(remade_graph, key, v, undirected)

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


def search_dask_graph(graph, proxy_to_slices, proxy_to_dict, origarr_to_used_proxies, origarr_to_obj, origarr_to_blocks_shape, unused_keys, main_components=None):
    """ Search proxies in the remade graph and fill in dictionaries to store information.
    """

    for key, v in graph.items():  
        print("\n", key)

        # if it is a subgraph, recurse
        if isinstance(v, dict):
            search_dask_graph(v, proxy_to_slices, proxy_to_dict, origarr_to_used_proxies, origarr_to_obj, origarr_to_blocks_shape, unused_keys, main_components)

        # if it is an original array, store it
        elif isinstance(key, str) and "array-original" in key:
            obj = v
            origarr_to_obj[key] = obj
            origarr_to_blocks_shape[key] = get_array_block_dims(obj.shape, obj.chunks)
            continue

        # if it is a task, add its arguments
        elif is_task(v) and (key not in unused_keys):  
            used_key = True
            if main_components:
                print("tested")
                used_key = False
                for main_comp in main_components:
                    if key in main_comp:
                        used_key = True 
                        print("found")
            if used_key:
                try:
                    f, target, slices = v
                    # search for values that are array-original, meaning that key is proxy 
                    if "array-original" in target and all([isinstance(s, slice) for s in slices]):
                        print("added")
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
        remade_graph = get_graph_from_dask(graph, undirected=True)
        unused_keys = list()
        connected_comps = BFS_connected_components(remade_graph,
                                          filter_condition_for_root_nodes=true_dumb_function,
                                          max_iterations=10)
        max_len = max(map(len, connected_comps.values()))
        main_components = [
            _list for comp,
            _list in connected_comps.items() if len(_list) == max_len]
        print("main_components", main_components)


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