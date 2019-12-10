import math
import sys
import dask
from dask.base import tokenize
import operator
from operator import getitem
from tests_utils import get_arr_shapes
from tests_utils import neat_print_graph, ONE_GIG
import optimize_io
from optimize_io.modifiers import add_to_dict_of_lists, get_config_chunk_shape
import logging


def apply_clustered_strategy(graph, dicts, chunk_shape):
    """ Main function applying clustered strategy on a dask graph.
    """
    for origarr_name in dicts['origarr_to_obj'].keys():
        if not origarr_name in dicts['origarr_to_used_proxies']:
            continue

        print("Creating buffers...")
        buffers = create_buffers(origarr_name, dicts, chunk_shape)
        print(f'Buffers scheduled: {buffers}')
        for buffer in buffers:            
            key = create_buffer_node(graph, origarr_name, dicts, buffer, chunk_shape)
            update_io_tasks(graph, dicts, buffer, key, chunk_shape)


def get_load_strategy(
        buffer_mem_size,
        cs,
        original_array_blocks_shape,
        nb=4):
    """ Get clustered writes best load strategy given the memory available for io optimization.

    block_row_size = block_mem_size * original_array_blocks_shape[2]
    block_slice_size = block_row_size * original_array_blocks_shape[1]

    Arguments: 
    ---------
        cs = chunk_shape
        nb = nb_bytes_per_val

    Returns:
    ---------
        strategy, 
        max_blocks_per_load
    """
    
    block_mem_size = cs[0] * cs[1] * cs[2] * nb
    strategy = "blocks" # for the moment, let the strategy be blocks only
    
    if (buffer_mem_size < block_mem_size):
        msg = "Not enough memory to store one block!"
        print(msg)
        raise ValueError(msg)
    max_blocks_per_load = math.floor(buffer_mem_size / block_mem_size)

    logging.debug(f'Memory available: {buffer_mem_size}')
    logging.debug(f'Chunk_shape: {cs}')
    logging.debug(f'Block_mem_size: {block_mem_size}')
    logging.debug(f'Strategy: {strategy}')
    logging.debug(f'Max nb blocks per load: {max_blocks_per_load}')
    return strategy, max_blocks_per_load


def start_new_buffer(curr_buffer, b, prev_block, strategy, nb_blocks_per_row, max_nb_blocks_per_buffer):
    """ Utility function for buffering
    """
    # test basic stuff
    if len(curr_buffer) == max_nb_blocks_per_buffer:
        logging.debug("max nb blocks per buffer reached")
        return True
    if prev_block and prev_block != b - 1:
        logging.debug("blocks non contiguous")
        return True 

    # test for bad configurations
    if strategy == "blocks": # for the moment: blocks strategy only 

        # if it is a last element of row
        if (prev_block + 1) % nb_blocks_per_row == 0: # because 0 % k = 0
            logging.debug(f"prev block is last element of row. nb_blocks_per_row: {nb_blocks_per_row}. (prev_block + 1): {(prev_block + 1)}")
            # (prev_block + 1) because our block indices start at 0
            return True

    """if strategy == rows: # for the moment: blocks strategy only 
        don’t concatenate two rows overlapping slices"""

    # else 
    return False


def overlap_slice(curr_buff, buff, blocks_shape):
    """ Utility function for buffering 

    curr_buff: current buffer containing 1+ entire rows
    buff: buffer containing a entire row
    """
    end_of_buffer = curr_buff[-1]
    start_of_row = buff[0]
    i_1 = numeric_to_3d_pos(end_of_buffer, blocks_shape, order='C')[0]
    i_2 = numeric_to_3d_pos(start_of_row, blocks_shape, order='C')[0]
    if i_1 != i_2:
        return True
    return False


def merge_rows(buffers, blocks_shape, nb_blocks_per_row, max_blocks_per_load):
    """ Utility function for buffering """
    merged_buffers = list()
    curr_buff = list()

    def is_complete_row(buff, nb_blocks_per_row):
        """we made sure that initial buffers contain only blocks from the same row
        therefore we can just see if number of blocks in buffer == nb blocks in a row"""
        return (len(buff) == nb_blocks_per_row)	

    logging.debug(f'\nBefore first concat: {buffers}')
    logging.debug(f'Max_blocks_per_load: {max_blocks_per_load}')
    for buff in buffers:
        logging.debug(f'Treating buff {buff}')
        if not is_complete_row(buff, nb_blocks_per_row):
            merged_buffers.append(buff) # dont process
        else:
            start_new_buffer = False 
            
            if len(curr_buff) > 0 and overlap_slice(curr_buff, buff, blocks_shape): # we dont want to overlap slices
                logging.debug("Overlaping")
                start_new_buffer = True 

            if len(curr_buff) + len(buff) > max_blocks_per_load:
                logging.debug("Nb max reached")
                start_new_buffer = True
            
            if start_new_buffer:
                logging.debug("Starting new buffer")
                merged_buffers.append(curr_buff)
                curr_buff = buff
            else:
                curr_buff = curr_buff + buff
    if len(curr_buff) > 0:  # add the last one
        logging.debug(f'Last buffer: {curr_buff}')
        merged_buffers.append(curr_buff)
    
    logging.debug(f'\nAfter first concat: {merged_buffers}')
    return merged_buffers


def merge_slices(merged_buffers, nb_blocks_per_slice, max_blocks_per_load):
    prev_slice = (None, None) # index, list of blocks
    merged_buffers_2 = list()
    curr_buff = list() # buffer to do the merge

    def is_complete_slice(buff, nb_blocks_per_slice):
        """
            we made sure 
                -that only contiguous blocks are in same buffer
                -rows overlaping slices have not been merged 
        """
        if not (len(buff) == nb_blocks_per_slice):
            return False, None
        else:
            return True, math.floor(buff[-1] / nb_blocks_per_slice)

    logging.debug(f'\nBefore slices concat: merged_buffers')
    for buff in merged_buffers:
        logging.debug(f'Treating: {buff}')
        is_slice, slice_index = is_complete_slice(buff, nb_blocks_per_slice)
        logging.debug(f'Slice_index: {slice_index}')
        if not is_slice:
            merged_buffers_2.append(buff)
        else:
            prev_slice_index = prev_slice[0]
            if not prev_slice_index == None:
                if slice_index == (prev_slice_index + 1):
                    if len(curr_buff) + len(buff) <= max_blocks_per_load:
                        logging.debug(f'len(curr_buff) + len(buff): {len(curr_buff) + len(buff)}')
                        logging.debug(f'VS max_blocks_per_load: {max_blocks_per_load}')
                        curr_buff = curr_buff + buff
                    else:
                        logging.debug("Creating new buffer0")
                        merged_buffers_2.append(curr_buff)
                        curr_buff = buff
                else:
                    logging.debug("Creating new buffer1")
                    merged_buffers_2.append(curr_buff)
                    curr_buff = buff
            else:
                logging.debug("Creating new buffer2")
                curr_buff = buff
            prev_slice = (slice_index, buff)
    if len(curr_buff) > 0:
        merged_buffers_2.append(curr_buff)

    logging.debug(f'\nAfter slices concat: {merged_buffers_2}')
    return merged_buffers_2

def create_buffers(origarr_name, dicts, chunk_shape):
    """ Merge used blocks into buffers following the 'clustered reads' strategy.
    """

    def get_buffer_size(default_memory=ONE_GIG):
        try:
            optimization = dask.config.get("io-optimizer")
            return dask.config.get("io-optimizer.memory_available")
        except BaseException:
            return default_memory

    def new_list(list_of_lists):
        list_of_lists.append(list())
        return list_of_lists, None

    # get strategy to apply
    blocks_shape = dicts['origarr_to_blocks_shape'][origarr_name] # WARNING: TODO change var name -> blocks_shape is origarr_to_blocks_shape
    strategy, max_nb_blocks_per_buffer = get_load_strategy(get_buffer_size(), 
                                                      chunk_shape, # get_config_chunk_shape(), 
                                                      blocks_shape)
                                                      
    # get the blocks used list to be bufferized
    arr_obj = dicts['origarr_to_obj'][origarr_name]
    blocks_used, block_to_proxies = get_blocks_used(dicts, origarr_name, arr_obj, chunk_shape)
    dicts['block_to_proxies'] = block_to_proxies
    blocks_used = sorted(blocks_used)

    # buffering part
    logging.debug(f'\nBefore creating buffers; blocks used are:{blocks_used}')
    buffers = buffering(blocks_used, strategy, blocks_shape, max_nb_blocks_per_buffer, True, True)
    return buffers


def buffering(blocks, strategy, blocks_shape, max_nb_blocks_per_buffer, row_concat=True, slices_concat=True):
    nb_blocks_per_row = blocks_shape[2]
    nb_blocks_per_slice = blocks_shape[2] * blocks_shape[1]
    
    buffers = list()
    curr_buff = list()
    prev_block = -1
    while len(blocks) > 0:
        b = blocks.pop(0)
        logging.debug(f'Treating block {b}')

        # prev_block >= 0 because remember 0 == False but we want to enter loop starting with second block (1st block index = 0)
        if prev_block >= 0 and start_new_buffer(curr_buff, b, prev_block, strategy, nb_blocks_per_row, max_nb_blocks_per_buffer):
            logging.debug("Starting a new buffer...")
            buffers.append(curr_buff.copy())
            curr_buff = list()
            prev_block = -1

        curr_buff.append(b)
        prev_block = b
    if len(curr_buff) > 0:
        buffers.append(curr_buff)

    # concatenation part (to be removed): Concat if complete rows/slices 
    if row_concat:
        buffers = merge_rows(buffers, blocks_shape, nb_blocks_per_row, max_nb_blocks_per_buffer) # 1) merge complete rows together
    if slices_concat:
        buffers = merge_slices(buffers, nb_blocks_per_slice, max_nb_blocks_per_buffer) # 2) merge complete slices together
    
    logging.debug(f'Final buffers scheduled:: {buffers}')
    return buffers


def get_blocks_used(dicts, origarr_name, arr_obj, chunk_shape):
    # extrapolate blocks to load (here talking of logical blocks)
    blocks_seen = list()
    blocks_used = list()
    block_to_proxies =  dict()
    used_proxies = dicts['origarr_to_used_proxies'][origarr_name]
    blocks_shape = dicts['origarr_to_blocks_shape'][origarr_name]
    for proxy_key in used_proxies:
        slice_tuple = dicts['proxy_to_slices'][proxy_key]
        logging.debug(f'slice_tuple found {slice_tuple}')        
        x_range, y_range, z_range = get_covered_blocks(slice_tuple, chunk_shape) # get_config_chunk_shape())
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    if (x, y, z) not in blocks_seen:
                        blocks_seen.append((x, y, z))
                        num_pos = _3d_to_numeric_pos((x, y, z), blocks_shape, 'C')
                        logging.debug(f'Associated {num_pos} to {(x, y, z)} using block shape: {blocks_shape}')
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
        buffer,
        chunk_shape):

    # get new key
    # create name in the form : '53c92348ec58571124ec14b40bc42677-merged'
    buffers_key = origarr_name.split('-')[-1] + '-merged'
    key = (buffers_key, buffer[0], buffer[-1])

    # get new value
    arr_obj = dicts['origarr_to_obj'][origarr_name]
    blocks_shape = dicts['origarr_to_blocks_shape'][origarr_name]
    buffer_slices = get_buffer_slices_from_original_array(buffer, blocks_shape, chunk_shape) # get_config_chunk_shape())
    value = (getitem, origarr_name, buffer_slices)
    logging.debug(f'Buffer_slices found: {buffer_slices}')

    # add new key/val pair to the dask graph
    if buffers_key in dask_graph.keys():
        dask_graph[buffers_key].update({key: value})
    else:
        dask_graph[buffers_key] = {key: value}
    return key


def update_io_tasks(graph, dicts, buffer, buffer_key, chunk_shape):
    for block_id in buffer:
        proxies = dicts['block_to_proxies'][block_id]
        for proxy in proxies:
            source_dict = dicts['proxy_to_dict'][proxy]
            val = source_dict[proxy]

            if len(val) == 2:
                _, slices = source_dict[proxy]
                slices = origarr_to_buffer_slices(dicts, proxy, buffer_key, slices, chunk_shape)
                source_dict[proxy] = (getitem, buffer_key, slices)

            elif len(val) == 3:
                _, _, slices = source_dict[proxy]
                slices = origarr_to_buffer_slices(dicts, proxy, buffer_key, slices, chunk_shape)
                source_dict[proxy] = (getitem, buffer_key, slices)

            else:
                # print("did nothing", len(val))
                pass


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


def origarr_to_buffer_slices(dicts, proxy, buffer_key, slices, chunk_shape):
    buffer_id, _ = buffer_key[0].split('-')
    origarr_name = 'array-original' + '-' + buffer_id
    origarr_obj = dicts['origarr_to_obj'][origarr_name]
    img_nb_blocks_per_dim = dicts['origarr_to_blocks_shape'][origarr_name]

    block_id, start_block, end_block = buffer_key
    start_pos = numeric_to_3d_pos(start_block, img_nb_blocks_per_dim, 'C')
    offset = [x * i for x, i in zip(start_pos, chunk_shape)]# get_config_chunk_shape())]

    new_slices = list()
    for i, s in enumerate(slices):
        start = s.start - offset[i]
        stop = s.stop - offset[i]
        new_slices.append(slice(start, stop, s.step))
    slices = (new_slices[0], new_slices[1], new_slices[2])
    return slices


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
