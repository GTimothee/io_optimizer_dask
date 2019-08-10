import sys


sys.path.insert(0,'/home/user/Documents/workspace/projects/samActivities') 
sys.path.insert(0,'/home/user/Documents/workspace/projects/dask') # custom dask
sys.path.insert(1,'/home/user/Documents/workspace/projects/samActivities/tests/tests') 
sys.path.insert(2,'/home/user/Documents/workspace/projects/samActivities/tests/optimize_io')


import time, os
import numpy as np
import math
import h5py
import dask
import dask.array as da


import experience3
import main


from utils import *
from test_get_dicts import *
from test_get_slices import *
from test_clustered import *
from test_main import *


def test_get_slices():
    print("testing get_slices")
    test_add_or_create_to_list_dict()
    test_get_keys_from_graph()
    test_get_rechunk_subkeys()
    test_test_source_key()
    test_get_slices_from_rechunk_subkeys()
    test_get_slices_from_rechunk_keys()
    test_get_slices_from_getitem_subkeys()
    test_get_slices_from_getitem_keys()
    test_get_slices_from_dask_graph()
    test_get_used_getitems_from_graph()
    test_BFS_connected_components()


def test_get_dicts():
    print("testing get_dicts")
    test_get_array_block_dims()
    test_get_original_array_from_proxy_array_name()
    test_get_arrays_dictionaries()


def test_clustered():
    print("testing clustered")
    test_convert_proxy_to_buffer_slices()
    test_add_getitem_task_in_graph()
    test_recursive_search_and_update()
    test_update_io_tasks_getitem()
    test_update_io_tasks_rechunk()
    test_update_io_tasks()
    test_create_buffer_node()
    test_create_buffers()
    test_is_in_load()
    test_get_buffer_slices_from_original_array()

def test_main_funcs():
    print("testing main funcs")
    test_sum()
    test_graph_verifier()
    test_convert_slices_list_to_numeric_slices()
    test_main()
    test_in_custom_dask()
    
def test_all():
    test_get_slices()
    test_get_dicts()
    test_clustered()
    test_main_funcs()

test_in_custom_dask(visuals=False, non_opti=False)