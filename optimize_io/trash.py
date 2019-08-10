## utilities (to print and/or visualize)
"""

def get_rechunk_keys_lists_from_dask_array(graph, printer=False):
    
    #(k, rechunk_merge_key) : k is the key inside the merge graph, rechunk_merge_key is the key of the merge graph in the whole graph
    
    def get_rechunkkeys_and_proxyarraykeys_from_dask_graph():
        proxy_array_keys = list()
        rechunk_merge_keys = list()
        for key in keys:
            if "array" in key:
                proxy_array_keys.append(key)
            elif "rechunk-merge" in key:
                rechunk_merge_keys.append(key)
            else:
                others.append(key)
        return proxy_array_keys, rechunk_merge_keys
    
    keys = list(graph.keys())
    others = list()
    
    
    proxy_array_keys, rechunk_merge_keys = get_rechunkkeys_and_proxyarraykeys_from_dask_graph()
    if len(rechunk_merge_keys) == 0:
        raise ValueError("no rechunk in this graph")

    rechunk_keys = list()
    for key in rechunk_merge_keys:
        rechunk_graph = graph[key]
        for k in list(rechunk_graph.keys()):
            if "rechunk-split" in k[0]:
                rechunk_keys.append((k, key))
        
    if printer:
        print(rechunk_keys)
        print(proxy_array_keys)
        print(others)
    return rechunk_keys, rechunk_graph, proxy_array_keys, others


def get_getitem_keys_lists_from_dask_array(graph, printer=False):
    keys = list(graph.keys())
    getitem_keys = list()
    proxy_array_keys = list()
    actions = list()

    for k in keys:
        if "array" in k:
            proxy_array_keys.append(k)
        elif "getitem" in k:
            getitem_keys.append(k)
        else:
            actions.append(k)

    if printer:
        print(getitem_keys)
        print(proxy_array_keys)
        print(actions)
    return getitem_keys, proxy_array_keys, actions

"""

"""def print_array_parts(graph, display_real_slices=False):
    
    #if not display_real_slices: print range for x, range for y, range for z
    #if display_real_slices: print the slices used in the program without reformating before printing
    
    getitem_keys, _, _ = get_getitem_keys_lists_from_dask_array(graph)

    getitem_supertasks = dict()
    for key in getitem_keys:
        getitem_graph = graph[key]
        getitem_parts = list(getitem_graph.keys())
        getitem_content = dict()
        for getitem_part in getitem_parts:
            value = getitem_graph[getitem_part]
            proxy_array_name = value[1][0]
            proxy_array_part = tuple(value[1][1:])
            slice_from_array_part = value[2]

            if display_real_slices:
                part_x = [s.start for s in slice_from_array_part]
                part_y = [s.stop for s in slice_from_array_part]
                part_z = [s.step for s in slice_from_array_part]

                slice_from_array_part = (part_x, part_y, part_z)

            if proxy_array_name in list(getitem_content.keys()):
                getitem_content[proxy_array_name].append((proxy_array_part, slice_from_array_part))
            else:
                getitem_content[proxy_array_name]= [(proxy_array_part, slice_from_array_part)]
        getitem_supertasks[key] = getitem_content

    for task_key in list(getitem_supertasks.keys()):
        print(task_key)
        task = getitem_supertasks[task_key]
        for array in list(task.keys()):
            print("\t", array)
            for e in task[array]:
                print("\t\t", e)
            print("\n")
        print("\n")
    return getitem_supertasks
"""

def print_dask_graph_first_layer(dask_graph):
    """ print the first layer of the dask graph including keys and associated values in a clear way
    """
    print("dask_graph")
    for key, val in dask_graph.items():
        if not 'array-' in key:
            print("key", key, "\n")
            print("val", val, "\n\n")