import math
from experience3.utils import create_random_cube, load_array_parts, get_dask_array_from_hdf5


def logical_chunks_tests(arr, case):
    if case == 'slabs_dask_interpol':
        slab_width = 6
        new_chunks_shape = (slab_width, arr.shape[1], arr.shape[2])
        nb_chunks = [math.floor(arr.shape[0] / slab_width), 1, 1]
        arr = arr[:slab_width * nb_chunks[0],:,:]
        arr = arr.rechunk((tuple([slab_width] * nb_chunks[0]),(1210),(1400)))
    elif case == 'slabs_previous_exp':
        slab_width = 192
        new_chunks_shape = (slab_width, arr.shape[1], arr.shape[2])
        nb_chunks = [math.floor(arr.shape[0] / slab_width), 1, 1]
        arr = arr[:slab_width * nb_chunks[0],:,:]
        arr = arr.rechunk((tuple([slab_width] * nb_chunks[0]),(1210),(1400)))
    elif case == 'blocks_dask_interpol':
        new_chunks_shape = tuple([c[0] for c in arr.chunks])
        nb_chunks = [len(c) for c in arr.chunks]
    elif case == 'blocks_previous_exp':
        new_chunks_shape = (770,605,700)
        nb_chunks = [int(arr.shape[i] / new_chunks_shape[i]) for i in range(3)]
        arr = arr.rechunk((tuple([new_chunks_shape[0]] * nb_chunks[0]),
                          tuple([new_chunks_shape[1]] * nb_chunks[1]),
                          tuple([new_chunks_shape[2]] * nb_chunks[2])))
    else:
        raise ValueError("error")
    print("new chunks", arr.shape, arr.chunks)
    print("new_chunks_shape", new_chunks_shape)

    all_arrays = list()
    for i in range(nb_chunks[0]):
        for j in range(nb_chunks[1]):
            for k in range(nb_chunks[2]):
                all_arrays.append(load_array_parts(arr=arr,
                                                   geometry="right_cuboid",
                                                   shape=new_chunks_shape,
                                                   upper_corner=(i * new_chunks_shape[0],
                                                                 j * new_chunks_shape[1],
                                                                 k * new_chunks_shape[2]),
                                                   random=False))

    # to del:
    all_arrays = all_arrays[:2]
    a5 = all_arrays.pop(0)
    for a in all_arrays:
        a5 = a5 + a
    return a5

def get_graph_for_tests(i):
    data_path = '/home/user/Documents/workspace/projects/samActivities/experience3/tests/data/bbsamplesize.hdf5'
    one_gig = 1000000000
    key = "data"
    arr = get_dask_array_from_hdf5(data_path, key)
    a5 = logical_chunks_tests(arr, cases[i])
    graph = a5.dask.dicts
    return graph

def neatly_print_dict(d):
    for k, v in d.items():
        print(k, v, "\n")

def get_rechunk_dict_without_proxy_array_sample():
    return {('rechunk-merge-bcfb966a39aa5079f6457f1530dd85df',
    0,
    0,
    0): ("<function dask.array.core.concatenate3(arrays)>", [[[('rechunk-split-bcfb966a39aa5079f6457f1530dd85df',
        0)],
        [('rechunk-split-bcfb966a39aa5079f6457f1530dd85df', 1)],
        [('rechunk-split-bcfb966a39aa5079f6457f1530dd85df', 2)]]]),
    ('rechunk-merge-bcfb966a39aa5079f6457f1530dd85df',
    0,
    0,
    1): ("<function dask.array.core.concatenate3(arrays)>", [[[('rechunk-split-bcfb966a39aa5079f6457f1530dd85df',
        3)],
        [('rechunk-split-bcfb966a39aa5079f6457f1530dd85df', 4)],
        [('rechunk-split-bcfb966a39aa5079f6457f1530dd85df', 5)]]]),
    ('rechunk-split-bcfb966a39aa5079f6457f1530dd85df',
    0): ("<function _operator.getitem(a, b, /)>", ('rechunk-merge-1ea6569403cc56ee4e2ec69a35dc1d1d',
    0,
    0,
    0), (slice(0, 60, None), slice(0, 403, None), slice(0, 700, None))),
    ('rechunk-split-bcfb966a39aa5079f6457f1530dd85df',
    1): ("<function _operator.getitem(a, b, /)>", ('rechunk-merge-1ea6569403cc56ee4e2ec69a35dc1d1d',
    0,
    1,
    0), (slice(0, 60, None), slice(0, 403, None), slice(0, 700, None))),
    ('rechunk-split-bcfb966a39aa5079f6457f1530dd85df',
    2): ("<function _operator.getitem(a, b, /)>", ('rechunk-merge-1ea6569403cc56ee4e2ec69a35dc1d1d',
    0,
    2,
    0), (slice(0, 60, None), slice(0, 404, None), slice(0, 700, None))),
    ('rechunk-split-bcfb966a39aa5079f6457f1530dd85df',
    3): ("<function _operator.getitem(a, b, /)>", ('rechunk-merge-1ea6569403cc56ee4e2ec69a35dc1d1d',
    0,
    0,
    1), (slice(0, 60, None), slice(0, 403, None), slice(0, 700, None))),
    ('rechunk-split-bcfb966a39aa5079f6457f1530dd85df',
    4): ("<function _operator.getitem(a, b, /)>", ('rechunk-merge-1ea6569403cc56ee4e2ec69a35dc1d1d',
    0,
    1,
    1), (slice(0, 60, None), slice(0, 403, None), slice(0, 700, None))),
    ('rechunk-split-bcfb966a39aa5079f6457f1530dd85df',
    5): ("<function _operator.getitem(a, b, /)>", ('rechunk-merge-1ea6569403cc56ee4e2ec69a35dc1d1d',
    0,
    2,
    1), (slice(0, 60, None), slice(0, 404, None), slice(0, 700, None)))
    }
    
def get_rechunk_dict_from_proxy_array_sample(array_name='array-3ec4eddf5e385f67eb8007734372b503', array_names=None, add_list=[1,2,3]):
    if array_names == None:
        array_names = [array_name, array_name, array_name]
    
    add1 = {('rechunk-split-a168f56ba79513b9ed87b2f22dd07458',
    3): ("<function _operator.getitem(a, b, /)>", (array_names[0],
    0,
    0,
    3), (slice(0, 220, None), slice(0, 242, None), slice(0, 100, None))),
    ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458',
    7): ("<function _operator.getitem(a, b, /)>", (array_names[0],
    0,
    1,
    3), (slice(0, 220, None), slice(0, 242, None), slice(0, 100, None)))}

    add2 = {('rechunk-split-a168f56ba79513b9ed87b2f22dd07458',
    8): ("<function _operator.getitem(a, b, /)>", (array_names[1],
    0,
    2,
    0), (slice(0, 220, None), slice(0, 121, None), slice(0, 200, None)))}


    add3 = {('rechunk-split-a168f56ba79513b9ed87b2f22dd07458',
    9): ("<function _operator.getitem(a, b, /)>", (array_names[2],
    0,
    2,
    1), (slice(0, 220, None), slice(0, 121, None), slice(0, 200, None))),
    ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458',
    10): ("<function _operator.getitem(a, b, /)>", (array_names[2],
    0,
    2,
    2), (slice(0, 220, None), slice(0, 121, None), slice(0, 200, None)))}


    d = {('rechunk-merge-a168f56ba79513b9ed87b2f22dd07458',
    0,
    0,
    0): ("<function dask.array.core.concatenate3(arrays)>", [[[(array_name, 0, 0, 0),
        (array_name, 0, 0, 1),
        (array_name, 0, 0, 2),
        ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 3)],
        [(array_name, 0, 1, 0),
        (array_name, 0, 1, 1),
        (array_name, 0, 1, 2),
        ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 7)],
        [('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 8),
        ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 9),
        ('rechunk-split-a168f56ba79513b9ed87b2f22dd07458', 10)]]])}

    if 1 in add_list:
        d.update(add1)
    if 2 in add_list:
        d.update(add2)
    if 3 in add_list:
        d.update(add3)
    return d

def get_getitem_dict_from_proxy_array_sample():
    return {('getitem-c6555b775be6a9d771866321a0d38252',
    0,
    0,
    0): ("<function _operator.getitem(a, b, /)>", ('array-6f870a321e8529128cb9bb82b8573db5',
    0,
    0,
    0), (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None))),
    ('getitem-c6555b775be6a9d771866321a0d38252',
    0,
    0,
    1): ("<function _operator.getitem(a, b, /)>", ('array-6f870a321e8529128cb9bb82b8573db5',
    0,
    0,
    1), (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None))),
    ('getitem-c6555b775be6a9d771866321a0d38252',
    0,
    0,
    2): ("<function _operator.getitem(a, b, /)>", ('array-6f870a321e8529128cb9bb82b8573db5',
    0,
    0,
    2), (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None))),
    ('getitem-c6555b775be6a9d771866321a0d38252',
    0,
    0,
    3): ("<function _operator.getitem(a, b, /)>", ('array-6f870a321e8529128cb9bb82b8573db5',
    0,
    0,
    3), (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None))),
    ('getitem-c6555b775be6a9d771866321a0d38252',
    0,
    0,
    4): ("<function _operator.getitem(a, b, /)>", ('array-6f870a321e8529128cb9bb82b8573db5',
    0,
    0,
    4), (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None))),
    ('getitem-c6555b775be6a9d771866321a0d38252',
    0,
    0,
    5): ("<function _operator.getitem(a, b, /)>", ('array-6f870a321e8529128cb9bb82b8573db5',
    0,
    0,
    5), (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None)))}


def get_graph_with_getitem():
    return {'add-3899a5d2265f04839b7c64d88116dc55': "<dask.blockwise.Blockwise at 0x7f85a3391438>",
 'getitem-430f856c4196ad50518e167d79ffd894': {('getitem-430f856c4196ad50518e167d79ffd894',
   0,
   0,
   0): ("<function _operator.getitem(a, b, /)>",
   ('array-4d8aa96f6f06806aeb9a11b75751b175', 0, 0, 0),
   (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None))),
  ('getitem-430f856c4196ad50518e167d79ffd894',
   0,
   0,
   1): ("<function _operator.getitem(a, b, /)>", ('array-4d8aa96f6f06806aeb9a11b75751b175',
    0,
    0,
    1), (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None))),
  ('getitem-430f856c4196ad50518e167d79ffd894',
   0,
   0,
   2): ("<function _operator.getitem(a, b, /)>", ('array-4d8aa96f6f06806aeb9a11b75751b175',
    0,
    0,
    2), (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None))),
  ('getitem-430f856c4196ad50518e167d79ffd894',
   0,
   0,
   3): ("<function _operator.getitem(a, b, /)>", ('array-4d8aa96f6f06806aeb9a11b75751b175',
    0,
    0,
    3), (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None))),
  ('getitem-430f856c4196ad50518e167d79ffd894',
   0,
   0,
   4): ("<function _operator.getitem(a, b, /)>", ('array-4d8aa96f6f06806aeb9a11b75751b175',
    0,
    0,
    4), (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None))),
  ('getitem-430f856c4196ad50518e167d79ffd894',
   0,
   0,
   5): ("<function _operator.getitem(a, b, /)>", ('array-4d8aa96f6f06806aeb9a11b75751b175',
    0,
    0,
    5), (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None))),
  ('getitem-430f856c4196ad50518e167d79ffd894',
   0,
   0,
   6): ("<function _operator.getitem(a, b, /)>", ('array-4d8aa96f6f06806aeb9a11b75751b175',
    0,
    0,
    6), (slice(None, None, None),
    slice(None, None, None),
    slice(None, None, None)))},
 'array-4d8aa96f6f06806aeb9a11b75751b175': {('array-4d8aa96f6f06806aeb9a11b75751b175',
   0,
   0,
   0): ("<function dask.array.core.getter(a, b, asarray=True, lock=None)>",
   'array-original-4d8aa96f6f06806aeb9a11b75751b175',
   (slice(0, 220, None), slice(0, 242, None), slice(0, 200, None))),
  ('array-4d8aa96f6f06806aeb9a11b75751b175',
   0,
   0,
   1): ("<function dask.array.core.getter(a, b, asarray=True, lock=None)>", 'array-original-4d8aa96f6f06806aeb9a11b75751b175', (slice(0, 220, None),
    slice(0, 242, None),
    slice(200, 400, None))),
  ('array-4d8aa96f6f06806aeb9a11b75751b175',
   0,
   0,
   2): ("<function dask.array.core.getter(a, b, asarray=True, lock=None)>", 'array-original-4d8aa96f6f06806aeb9a11b75751b175', (slice(0, 220, None),
    slice(0, 242, None),
    slice(400, 600, None))),
  ('array-4d8aa96f6f06806aeb9a11b75751b175',
   0,
   0,
   3): ("<function dask.array.core.getter(a, b, asarray=True, lock=None)>", 'array-original-4d8aa96f6f06806aeb9a11b75751b175', (slice(0, 220, None),
    slice(0, 242, None),
    slice(600, 800, None))),
  ('array-4d8aa96f6f06806aeb9a11b75751b175',
   0,
   0,
   4): ("<function dask.array.core.getter(a, b, asarray=True, lock=None)>", 'array-original-4d8aa96f6f06806aeb9a11b75751b175', (slice(0, 220, None),
    slice(0, 242, None),
    slice(800, 1000, None))),
  ('array-4d8aa96f6f06806aeb9a11b75751b175',
   0,
   0,
   5): ("<function dask.array.core.getter(a, b, asarray=True, lock=None)>", 'array-original-4d8aa96f6f06806aeb9a11b75751b175', (slice(0, 220, None),
    slice(0, 242, None),
    slice(1000, 1200, None)))}}



concat_func = None
rechunk_merge_example = {('rechunk-merge-7c9f5c6cedeb992c5f39c40adfae384b',
  0,
  0,
  0): (concat_func, [[[('array-edb82dc9c0509fc1e17bc84538520340',
      0,
      0,
      0),
     ('array-edb82dc9c0509fc1e17bc84538520340', 0, 0, 1),
     ('array-edb82dc9c0509fc1e17bc84538520340', 0, 0, 2),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 3)],
    [('array-edb82dc9c0509fc1e17bc84538520340', 0, 1, 0),
     ('array-edb82dc9c0509fc1e17bc84538520340', 0, 1, 1),
     ('array-edb82dc9c0509fc1e17bc84538520340', 0, 1, 2),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 7)],
    [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 8),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 9),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 10),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 11)]],
   [[('array-edb82dc9c0509fc1e17bc84538520340', 1, 0, 0),
     ('array-edb82dc9c0509fc1e17bc84538520340', 1, 0, 1),
     ('array-edb82dc9c0509fc1e17bc84538520340', 1, 0, 2),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 15)],
    [('array-edb82dc9c0509fc1e17bc84538520340', 1, 1, 0),
     ('array-edb82dc9c0509fc1e17bc84538520340', 1, 1, 1),
     ('array-edb82dc9c0509fc1e17bc84538520340', 1, 1, 2),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 19)],
    [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 20),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 21),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 22),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 23)]],
   [[('array-edb82dc9c0509fc1e17bc84538520340', 2, 0, 0),
     ('array-edb82dc9c0509fc1e17bc84538520340', 2, 0, 1),
     ('array-edb82dc9c0509fc1e17bc84538520340', 2, 0, 2),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 27)],
    [('array-edb82dc9c0509fc1e17bc84538520340', 2, 1, 0),
     ('array-edb82dc9c0509fc1e17bc84538520340', 2, 1, 1),
     ('array-edb82dc9c0509fc1e17bc84538520340', 2, 1, 2),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 31)],
    [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 32),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 33),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 34),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 35)]],
   [[('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 36),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 37),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 38),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 39)],
    [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 40),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 41),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 42),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 43)],
    [('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 44),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 45),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 46),
     ('rechunk-split-7c9f5c6cedeb992c5f39c40adfae384b', 47)]]])}