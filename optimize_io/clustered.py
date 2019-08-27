import math
import sys

from dask.base import tokenize
import operator
from operator import getitem
from tests_utils import get_arr_shapes
from tests_utils import neat_print_graph
import optimize_io
from optimize_io.modifiers import add_to_dict_of_lists


def apply_clustered_strategy(graph, dicts):
    """ Main function applying clustered strategy on a dask graph.
    """
    for origarr_name in dicts['origarr_to_obj'].keys():
        buffers = create_buffers(origarr_name, dicts)
        
        for buffer in buffers:            
            key = create_buffer_node(graph, origarr_name, dicts, buffer)
            update_io_tasks(graph, dicts, buffer, key)


def get_load_strategy(
        buffer_mem_size,
        chunk_shape,
        original_array_blocks_shape,
        nb_bytes_per_val=4):
    """ get clustered writes best load strategy given 
    the memory available for io optimization
    """
    block_mem_size = chunk_shape[0] * \
        chunk_shape[1] * chunk_shape[2] * nb_bytes_per_val
    block_row_size = block_mem_size * original_array_blocks_shape[2]
    block_slice_size = block_row_size * original_array_blocks_shape[1]

    if buffer_mem_size >= block_slice_size:
        nb_slices = math.floor(buffer_mem_size / block_slice_size)
        return "slices", nb_slices * \
            original_array_blocks_shape[2] * original_array_blocks_shape[1]
    elif buffer_mem_size >= block_row_size:
        nb_rows = math.floor(buffer_mem_size / block_row_size)
        return "rows", nb_rows * original_array_blocks_shape[2]
    else:
        return "blocks", math.floor(buffer_mem_size / block_mem_size)


def create_buffers(origarr_name, dicts):

    def get_buffer_size(default_memory=1000000000):
        try:
            optimization = config.get("io-optimizer")
            try:
                return config.get("io-optimizer.memory_available")
            except BaseException:
                print("missing configuration information memory_available")
                print("using default configuration: 1 gigabytes")
                return default_memory
        except BaseException:
            return default_memory

    def new_list(list_of_lists):
        list_of_lists.append(list())
        return list_of_lists, None

    def bad_configuration_incoming(
            prev_i,
            strategy,
            original_array_blocks_shape):
        """ to avoid bad configurations in clustered writes
        """

        if not prev_i:
            return False
        elif strategy == "blocks" and (prev_i % original_array_blocks_shape[2] - 1) == 0:
            return True
        elif (strategy == "rows") and (prev_i > 1) and (
                ((prev_i + 1) % original_array_blocks_shape[2]) == 0):
            return True
        else:
            return False

    def test_if_create_new_load(
            list_of_lists,
            prev_i,
            strategy,
            original_array_blocks_shape):
        if len(list_of_lists[-1]) == max_blocks_per_load:
            return new_list(list_of_lists)
        elif prev_i and next_i != prev_i + 1:
            return new_list(list_of_lists)
        elif bad_configuration_incoming(prev_i, strategy, original_array_blocks_shape):
            return new_list(list_of_lists)
        else:
            return list_of_lists, prev_i


    # just get some information used later
    arr_obj = dicts['origarr_to_obj'][origarr_name]
    blocks_shape = dicts['origarr_to_blocks_shape'][origarr_name]
    strategy, max_blocks_per_load = get_load_strategy(get_buffer_size(), 
                                                      arr_obj.chunks, 
                                                      blocks_shape)
    #TODO: revoir stratégies et résultats des tests en conséquence
    print("strategy:", strategy)
    print("max nb blocks per load:", max_blocks_per_load)
    blocks_used, block_to_proxies = get_blocks_used(dicts, origarr_name, arr_obj)
    dicts['block_to_proxies'] = block_to_proxies

    # create buffers
    list_of_lists, prev_i = new_list(list())
    while len(blocks_used) > 0:
        next_i = blocks_used.pop(0)
        list_of_lists, prev_i = test_if_create_new_load(
            list_of_lists, prev_i, strategy, blocks_shape)
        list_of_lists[len(list_of_lists) - 1].append(next_i)
        prev_i = next_i

    return list_of_lists


def get_blocks_used(dicts, origarr_name, arr_obj):
    # extrapolate blocks to load (here talking of logical blocks)
    blocks_seen = list()
    blocks_used = list()
    block_to_proxies =  dict()
    used_proxies = dicts['origarr_to_used_proxies'][origarr_name]
    blocks_shape = dicts['origarr_to_blocks_shape'][origarr_name]
    for proxy_key in used_proxies:
        slice_tuple = dicts['proxy_to_slices'][proxy_key]        
        x_range, y_range, z_range = get_covered_blocks(slice_tuple, arr_obj.chunks)
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    if (x, y, z) not in blocks_seen:
                        blocks_seen.append((x, y, z))
                        num_pos = _3d_to_numeric_pos((x, y, z), blocks_shape, 'C')
                        blocks_used.append(num_pos)
                        block_to_proxies = add_to_dict_of_lists(block_to_proxies, num_pos, proxy_key, unique=True)
    return blocks_used, block_to_proxies


def get_covered_blocks(slice_tuple, chunk_shape):
    """ From list of slices of type (slice, slice, slice) find 
    which blocks of original array are used by this slicing.
    we are speaking of logical block here, hence the 'chunk_shape' parameter
    """
    ranges = list()
    for i, s in zip(range(3), slice_tuple):
        a = math.floor(s.start / chunk_shape[i])
        b = math.floor((s.stop - 1) / chunk_shape[i])  # (s.stop - 1) because begins at 0 not 1
        ranges.append(range(a, b + 1))  # b + 1 because we want all from a to b included

    return ranges


def create_buffer_node(
        dask_graph,
        origarr_name,
        dicts,
        buffer):

    # get new key
    # create name in the form : '53c92348ec58571124ec14b40bc42677-merged'
    buffers_key = origarr_name.split('-')[-1] + '-merged'
    key = (buffers_key, buffer[0], buffer[-1])

    # get new value
    arr_obj = dicts['origarr_to_obj'][origarr_name]
    buffer_slices = get_buffer_slices_from_original_array(buffer, arr_obj.shape, arr_obj.chunks)
    value = (getitem, origarr_name, buffer_slices)

    # add new key/val pair to the dask graph
    if buffers_key in dask_graph.keys():
        dask_graph[buffers_key].update({key: value})
    else:
        dask_graph[buffers_key] = {key: value}
    return key


def update_io_tasks(graph, dicts, buffer, buffer_key):
    for block_id in buffer:
        proxies = dicts['block_to_proxies'][block_id]
        for proxy in proxies:
            source_dict = dicts['proxy_to_dict'][proxy]
            val = source_dict[proxy]

            if len(val) == 2:
                _, slices = source_dict[proxy]
                origarr_to_buffer_slices(dicts, proxy, buffer_key, slices)
                source_dict[proxy] = (getitem, buffer_key, slices)

            elif len(val) == 3:
                _, _, slices = source_dict[proxy]
                origarr_to_buffer_slices(dicts, proxy, buffer_key, slices)
                source_dict[proxy] = (getitem, buffer_key, slices)

            else:
                print("did nothing", len(val))


def get_buffer_slices_from_original_array(load, shape, original_array_chunk):
    start = min(load)
    end = max(load)

    all_block_num_indexes = range(start, end + 1)
    all_block_3d_indexes = [
        numeric_to_3d_pos(
            num_pos,
            shape,
            order='C') for num_pos in all_block_num_indexes]

    mini = [None, None, None]
    maxi = [None, None, None]
    for _3d_index in all_block_3d_indexes:
        for i in range(3):
            if (mini[i] is None) or (_3d_index[i] < mini[i]):
                mini[i] = _3d_index[i]
            if maxi[i] is None or (_3d_index[i] + 1 > maxi[i]):
                maxi[i] = _3d_index[i] + 1

    mini = [e * d for e, d in zip(mini, original_array_chunk)]
    maxi = [e * d for e, d in zip(maxi, original_array_chunk)]

    return (slice(mini[0], maxi[0], None),
            slice(mini[1], maxi[1], None),
            slice(mini[2], maxi[2], None))


def origarr_to_buffer_slices(dicts, proxy, buffer_key, slices):

    buffer_id, _ = buffer_key[0].split('-')
    origarr_name = 'array-original' + '-' + buffer_id
    origarr_obj = dicts['origarr_to_obj'][origarr_name]
    img_nb_blocks_per_dim = dicts['origarr_to_blocks_shape'][origarr_name]
    num_start_of_buffer = buffer_key[1]

    block_id, start_block, end_block = buffer_key
    start_pos = numeric_to_3d_pos(start_block, origarr_obj.shape, 'C')
    offset = [x * i for x, i in zip(start_pos, origarr_obj.chunks)]

    new_slices = list()
    for i, s in enumerate(slices):
        start = s.start - offset[i]
        end = s.start - offset[i]
        new_slices.append(slice(start, end, s.step))
    slices = (new_slices[0], new_slices[1], new_slices[2])


def numeric_to_3d_pos(numeric_pos, blocks_shape, order):
    if order == 'F':
        nb_blocks_per_row = blocks_shape[0]
        nb_blocks_per_slice = blocks_shape[0] * blocks_shape[1]
    elif order == 'C':
        nb_blocks_per_row = blocks_shape[2]
        nb_blocks_per_slice = blocks_shape[1] * blocks_shape[2]
    else:
        raise ValueError("unsupported")

    i = math.floor(numeric_pos / nb_blocks_per_slice)
    numeric_pos -= i * nb_blocks_per_slice
    j = math.floor(numeric_pos / nb_blocks_per_row)
    numeric_pos -= j * nb_blocks_per_row
    k = numeric_pos
    return (i, j, k)


def _3d_to_numeric_pos(_3d_pos, shape, order):
    """
    in C order for example, should be ((_3d_pos[0]-1) * nb_blocks_per_slice)
    but as we start at 0 we can keep (_3d_pos[0] * nb_blocks_per_slice)
    """
    if order == 'F':
        nb_blocks_per_row = shape[0]
        nb_blocks_per_slice = shape[0] * shape[1]
    elif order == 'C':
        nb_blocks_per_row = shape[2]
        nb_blocks_per_slice = shape[1] * shape[2]
    else:
        raise ValueError("unsupported")

    return (_3d_pos[0] * nb_blocks_per_slice) + \
        (_3d_pos[1] * nb_blocks_per_row) + _3d_pos[2]
