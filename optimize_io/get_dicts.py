__all__ = ("get_arrays_dictionaries", "get_original_array_from_proxy_array_name", "get_array_block_dims")


def get_arrays_dictionaries(graph, slices_dict):
    """
        Fill in three utility dictionnaries that will be useful for the next steps
    """

    array_to_original = dict()
    original_array_chunks = dict()
    original_array_shapes = dict()
    original_array_blocks_shape = dict()

    for proxy_array_name in list(slices_dict.keys()):
        original_array_name, original_array_obj = get_original_array_from_proxy_array_name(graph, proxy_array_name)
        array_to_original[proxy_array_name] = original_array_name
        original_array_shapes[original_array_name] = original_array_obj.shape
        original_array_chunks[original_array_name] = original_array_obj.chunks
        original_array_blocks_shape[original_array_name] = get_array_block_dims(original_array_obj.shape, original_array_obj.chunks)
        
    return array_to_original, original_array_chunks, original_array_shapes, original_array_blocks_shape


def get_original_array_from_proxy_array_name(graph, proxy_array_name):
    proxy_dict = graph[proxy_array_name]
    for chunk_key in list(proxy_dict.keys()):
        if isinstance(chunk_key, str):
            if 'array-original' in chunk_key:
                original_array_name = chunk_key
                break 
    if not original_array_name:
        raise ValueError("Original array not found. Are you sure that you gave a proxy array?")
    return original_array_name, proxy_dict[original_array_name]


def get_array_block_dims(shape, chunks):
    """ from shape of image and size of chukns=blocks, return the dimensions of the array in terms of blocks
    i.e. number of blocks in each dimension
    """
    if not len(shape) == len(chunks):
        raise ValueError("chunks and shape should have the same dimension", shape, chunks)
    return tuple([int(s/c) for s, c in zip(shape, chunks)])


