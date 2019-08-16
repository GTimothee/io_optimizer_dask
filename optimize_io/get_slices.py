__all__ = (
    "get_slices_from_dask_graph",
    "get_slices_from_rechunk_keys",
    "get_slices_from_getitem_keys",
    "get_slices_from_rechunk_subkeys",
    "get_slices_from_getitem_subkeys",
    "get_slices_from_getitem_subkeys",
    "check_source_key",
    "get_rechunk_subkeys",
    "get_keys_from_graph",
    "add_to_dict_of_lists",
    "get_getitems_from_graph",
    "get_used_getitems_from_graph",
    "BFS_connected_components")


# TODO generalize it to a graph/tree search
def get_slices_from_dask_graph(graph, used_getitems):
    keys_dict = get_keys_from_graph(graph)
    slices_dict = dict()
    deps_dict = dict()
    
    if 'store' in list(keys_dict.keys()):
        hidden_getitems = list()
        unknown_keys = keys_dict['store']
        for k in unknown_keys:
            v = graph[k]
            for k2, v2 in v.items():
                if isinstance(k2, tuple) and isinstance(k2[0], str) and ('array-getitem' in k2[0]):
                    hidden_getitems.append(k)
                    continue
        if len(hidden_getitems) != 0:
           keys_dict.update({'getitem': hidden_getitems})

    if 'rechunk-merge' in list(keys_dict.keys()):
        rechunk_keys = keys_dict['rechunk-merge']
        s1, d1 = get_slices_from_rechunk_keys(
            graph, rechunk_keys)  # register the getitem used
        slices_dict.update(s1)
        deps_dict.update(d1)

    if 'getitem' in list(keys_dict.keys()):
        getitem_keys = keys_dict['getitem']
        # add a condition to just take those that are used in the graph
        s2, d2 = get_slices_from_getitem_keys(
            graph, getitem_keys, used_getitems)
        slices_dict.update(s2)
        deps_dict.update(d2)

    return slices_dict, deps_dict


# get slices from keys
def get_slices_from_rechunk_keys(graph, rechunk_keys):
    global_slices_dict = dict()
    global_deps_dict = dict()

    for rechunk_key in rechunk_keys:
        rechunk_graph = graph[rechunk_key]
        split_keys, merge_keys = get_rechunk_subkeys(rechunk_graph)
        local_slices_dict, local_deps_dict = get_slices_from_rechunk_subkeys(
            rechunk_graph, split_keys, merge_keys)
        global_slices_dict.update(local_slices_dict)
        global_deps_dict.update(local_deps_dict)

    return global_slices_dict, global_deps_dict


def get_slices_from_getitem_keys(graph, getitem_keys, used_getitems):
    global_slices_dict = dict()
    global_deps_dict = dict()
    for getitem_key in getitem_keys:
        getitem_graph = graph[getitem_key]
        local_slices_dict, local_deps_dict = get_slices_from_getitem_subkeys(
            getitem_graph, used_getitems)

        # slices
        for k, v in local_slices_dict.items():
            if k not in list(global_slices_dict.keys()):
                global_slices_dict.update(local_slices_dict)
            else:
                global_slices_dict[k] = global_slices_dict[k] + \
                    local_slices_dict[k]

        # dependencies
        for k, v in local_deps_dict.items():
            if k not in list(global_deps_dict.keys()):
                global_deps_dict.update(local_deps_dict)
            else:
                global_deps_dict[k] = global_deps_dict[k] + local_deps_dict[k]

    return global_slices_dict, global_deps_dict


# get slices from subkeys
def get_slices_from_rechunk_subkeys(
        rechunk_merge_graph,
        split_keys,
        merge_keys):

    def get_slices_from_splits(split_keys, slices_dict, deps_dict):

        for split_key in split_keys:
            split_value = rechunk_merge_graph[split_key]
            _, source_key, slices = split_value
            slices_dict, deps_dict = check_source_key(
                slices_dict, deps_dict, source_key, split_key)
        return slices_dict, deps_dict

    def recursive_search(_list, merge_key, slices_dict, deps_dict):
        if not isinstance(_list[0], tuple):  # if it is not a list of targets
            for i in range(len(_list)):
                sublist = _list[i]
                slices_dict, deps_dict = recursive_search(
                    sublist, merge_key, slices_dict, deps_dict)
        else:
            for i in range(len(_list)):
                target_key = _list[i]
                if 'array' in target_key[0] and 'array-original' not in target_key[0]:
                    slices_dict, deps_dict = check_source_key(
                        slices_dict, deps_dict, target_key, merge_key)

        return slices_dict, deps_dict

    def get_slices_from_merges2(
            rechunk_merge_graph,
            merge_keys,
            slices_dict,
            deps_dict):
        for merge_key in merge_keys:
            val = rechunk_merge_graph[merge_key]
            f, concat_list = val
            slices_dict, deps_dict = recursive_search(
                concat_list, merge_key, slices_dict, deps_dict)
        return slices_dict, deps_dict

    slices_dict = dict()
    deps_dict = dict()
    slices_dict, deps_dict = get_slices_from_splits(
        split_keys, slices_dict, deps_dict)

    slices_dict, deps_dict = get_slices_from_merges2(
        rechunk_merge_graph, merge_keys, slices_dict, deps_dict)
    return slices_dict, deps_dict


def get_slices_from_getitem_subkeys(getitem_graph, used_getitems):
    slices_dict = dict()
    deps_dict = dict()

    for k, v in getitem_graph.items():
        if len(v) == 3:
            f, source_key, s = v
        else:
            raise ValueError("not supported")
        if isinstance(k[0], str) and "getitem" in k[0] and k in used_getitems:
            slices_dict, deps_dict = check_source_key(
                slices_dict, deps_dict, source_key, k)
    return slices_dict, deps_dict


def check_source_key(slices_dict, deps_dict, source_key, dependent_key):
    """ test if source is an array proxy: if yes, add source key data to slices_dict
    dependent_key: key of the task dependent from array proxy
    """

    if len(source_key) != 4:
        raise ValueError("not enough elements to unpack in", source_key)
    if not isinstance(source_key, tuple):
        raise ValueError("expected a tuple:", source_key)

    source, s1, s2, s3 = source_key

    if not isinstance(source, str):
        raise ValueError("expected a string:", source)
    if 'array' in source and 'array-original' not in source:
        slices_dict = add_to_dict_of_lists(
            slices_dict, source, (s1, s2, s3))
            
        deps_dict = add_to_dict_of_lists(
            deps_dict, source, dependent_key)
    return slices_dict, deps_dict


# get subkeys
def get_rechunk_subkeys(rechunk_graph):
    keys_dict = get_keys_from_graph(rechunk_graph, printer=False)
    return keys_dict['rechunk-split'], keys_dict['rechunk-merge']


def true_dumb_function(x):
    return True


#TODO: verify
def BFS_connected_components(
        graph,
        filter_condition_for_root_nodes=true_dumb_function,
        max_iterations=10):
    """
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
    nodes = list(graph.keys())
    nb_nodes_total = len(nodes)
    visited = dict(zip(nodes, nb_nodes_total * [False]))
    nb_nodes_visited = 0

    # initialize component variables
    components = dict()
    component_id = 0

    max_iterations = 10
    nb_its = 0
    while all_nodes_not_visited(nb_nodes_visited, nb_nodes_total):

        # get next unvisited node (next start of connected component)
        node_queue = list()
        for n in nodes:
            if not visited[n]:
                if filter_condition_for_root_nodes(n):
                    nb_nodes_visited = visit_node(
                        visited, n, nb_nodes_visited, components, node_queue)
                    break

        # run BFS
        while len(node_queue) > 0:
            curr_node = node_queue.pop(0)
            for n in graph[curr_node]:
                if not visited[n]:
                    nb_nodes_visited = visit_node(
                        visited, n, nb_nodes_visited, components, node_queue)
        component_id += 1

        # print("it", nb_its)
        nb_its += 1
        if max_iterations == nb_its:
            break

    return components


def get_used_getitems_from_graph(graph, undirected):
    """ search in the graph the use of getitem tasks so that we just 
    take them into account to create the buffers of clustered strategy
    This function searches for connected components in the dask graph 
    and take the deeper/deepest one/ones.

    • remake graph into undirected graph
    • BFS on it
    • extract main components (those with maximum depth)
    • go through the main components and if node contains “getitem” then add it to list of used getitems 
    """
    def recursive_search(_list, neighbours_list):
        """ search in depth into a value of k,v pairs
        """
        if not isinstance(_list[0], tuple):  # if it is not a list of targets
            for i in range(len(_list)):
                sublist = _list[i]
                neighbours_list = recursive_search(sublist, neighbours_list)
        else:
            for i in range(len(_list)):
                target_key = _list[i]
                neighbours_list.append(target_key)
        return neighbours_list

    def get_remade_graph(graph, undirected=False):
        """ Transform dask graph into a real graph in order to use BFS on it.
        """
        remade_graph = dict()
        for k, v in graph.items():
            if isinstance(k, tuple):
                k2 = k[0]
            elif isinstance(k, str) or isinstance(k, int):
                k2 = k
            else:
                raise ValueError("type of key unsupported", k, type(k))

            for k2, v2 in v.items():
                if isinstance(v2, str):
                    add_to_dict_of_lists(
                        remade_graph, k2, v2, unique=True)
                    if undirected:
                        add_to_dict_of_lists(
                            remade_graph, v2, k2, unique=True)
                    else:
                        add_to_dict_of_lists(
                            remade_graph, v2, None, unique=True)
                elif isinstance(v2, tuple):
                    if len(v2) > 1:
                        for arg in v2[1:]:
                            if isinstance(arg, list):
                                neighbours_list = recursive_search(arg, list())
                                for n in neighbours_list:
                                    add_to_dict_of_lists(
                                        remade_graph, k2, n, unique=True)
                                    if undirected:
                                        add_to_dict_of_lists(
                                            remade_graph, n, k2, unique=True)
                                    else:
                                        add_to_dict_of_lists(
                                            remade_graph, n, None, unique=True)

                            elif isinstance(arg, tuple) and isinstance(arg[0], str):
                                add_to_dict_of_lists(
                                    remade_graph, k2, arg, unique=True)
                                if undirected:
                                    add_to_dict_of_lists(
                                        remade_graph, arg, k2, unique=True)
                                else:
                                    add_to_dict_of_lists(
                                        remade_graph, arg, None, unique=True)

        """for k, v in remade_graph.items():
            print("\n\n", k)
            print("\n", v)"""

        return remade_graph

    def func(n):
        if isinstance(n, tuple) and 'getitem' in n[0]:
            return True
        return False

    remade_graph = get_remade_graph(graph, undirected=undirected)
    connected_comps = BFS_connected_components(
        remade_graph, filter_condition_for_root_nodes=func)
    max_len = max(map(len, connected_comps.values()))
    main_components = [
        _list for comp,
        _list in connected_comps.items() if len(_list) == max_len]

    """import sys
    for c in main_components:
        print("\n\n", c)
    sys.exit()"""

    used_getitems = list()
    for main_comp in main_components:
        for e in main_comp:
            if isinstance(
                    e,
                    tuple) and 'getitem' in e[0] and e not in used_getitems:
                used_getitems.append(e)

    return used_getitems


def get_getitems_from_graph(graph):
    """ not useful, use it if get_used_getitems_from_graph does not work just to verify that the results are corrects
    search in the graph the use of getitem tasks so that we just take them into account to create the buffers of clustered strategy
    """
    def recursive_search(_list, used_getitems):
        if not isinstance(_list[0], tuple):  # if it is not a list of targets
            for i in range(len(_list)):
                sublist = _list[i]
                used_getitems = recursive_search(sublist, used_getitems)
        else:
            for i in range(len(_list)):
                target_key = _list[i]
                if isinstance(target_key,
                              tuple) and 'getitem' in target_key[0]:
                    used_getitems.append(target_key)
        return used_getitems

    used_getitems = list()
    for k, v in graph.items():
        if isinstance(k, tuple):
            k2 = k[0]
        elif isinstance(k, str):
            k2 = k
        else:
            raise ValueError("type of key unsupported", k, type(k))

        split = k2.split('-')
        key_name = "-".join(split[:-1])  # take all but the ID
        if not key_name == "getitem":
            for subkey, subval in v.items():
                if isinstance(subval, tuple):
                    args = subval[1:]
                    for arg in args:
                        if isinstance(
                                arg, tuple) and isinstance(
                                arg[0], str) and 'getitem' in arg[0]:
                            used_getitems.append(arg)
                        elif isinstance(arg, list):
                            used_getitems = recursive_search(
                                arg, used_getitems)
    return used_getitems


# get keys
def get_keys_from_graph(graph, printer=False):
    key_dict = dict()
    for k, v in graph.items():
        if isinstance(k, tuple):
            k2 = k[0]
        elif isinstance(k, str):
            k2 = k
        elif isinstance(k, int): # for the moment just store taken into account
            key_name = 'store'
            key_dict = add_to_dict_of_lists(key_dict, key_name, k)
            continue 
        else:
            raise ValueError("type of key unsupported", k, type(k))
        split = k2.split('-')
        key_name = "-".join(split[:-1])
        key_dict = add_to_dict_of_lists(key_dict, key_name, k)
    return key_dict


# utility
def add_to_dict_of_lists(d, k, v, unique=False):
    """ if key does not exist, add a new list [value], else, 
    append value to existing list corresponding to the key
    """
    if k not in d:
        if v:
            d[k] = [v]
        else:
            d[k] = []
    else:
        if v and (unique and v not in d[k]) or not unique:
            d[k].append(v)
    return d
