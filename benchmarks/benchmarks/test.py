import os
import traceback
from tests_utils import CaseConfig

ONE_GIG = 1000000000

class Test():
    def __init__(self, test_params):
        self.hardware = test_params[0]
        self.cube_type = test_params[1]
        self.physik_chunked = test_params[2]
        self.chunk_type = test_params[3]
        self.scheduler_opti = test_params[4]
        self.opti = test_params[5]
        self.cube_ref = self.get_cube_ref()
        self.cube_shape = self.get_cube_shape()
        self.chunks_shape = self.get_chunks_shape()

        # create dask config
        self.array_filepath = os.path.join(self.get_hardware_path(), str(self.get_cube_ref()) + '.hdf5')
        split_file_path = os.path.join(self.get_hardware_path(), "split.hdf5")

        chunks_shape = self.get_chunks_shape()
        buffer_size = self.get_buffer_size()
        try:
            self.dask_config = CaseConfig(self.array_filepath, chunks_shape)
            self.dask_config.optimization(self.opti, self.scheduler_opti, buffer_size)
            self.dask_config.split_case(self.array_filepath, split_file_path)
        except Exception as e:
            print(traceback.format_exc())
            print("Something went wrong while creating dask config.")
            exit(1)
        return     


    def get_hardware_path(self):
        hardware_paths = {  # input and output dir
            'ssd': os.getenv('SSD_PATH'),  
            'hdd': os.getenv('HDD_PATH')  
        }
        return hardware_paths[self.hardware]

    def get_chunks_shape(self):
        chunks_shapes = {
            "very_small":{
                "blocks":[[200,200,200]],
                "slabs":[[50, 400, 400]]
            },
            "small":{
                "blocks":[
                    [700, 700, 700]],
                "slabs":[
                    [1400, 1400, "auto"],
                    [1400, 1400, 5],
                    [1400, 1400, 175]]
            },
            "big":{
                "blocks":[
                    [350, 350, 350],
                    [600, 600, 600],
                    [875, 875, 875],
                    [1750, 1750, 1750]],
                "slabs":[
                    [3500, 3500, "auto"],
                    [3500, 3500, 1],
                    [3500, 3500, 28],
                    [3500, 3500, 50],
                    [3500, 3500, 500]]
            }
        }
        return chunks_shapes[self.cube_type][self.chunk_type]

    def get_buffer_size(self):
        buffer_sizes = {
            "very_small": ONE_GIG,
            "small": 5.5 * ONE_GIG,
            "big": 15
        }
        return buffer_sizes[self.cube_type]


    def get_cube_ref(self):
        # chunked mean physical chunk here
        chunk_status = "physik_chunked" if self.physik_chunked else "not_physik_chunked"
        cube_refs = {
            'small': {
                'physik_chunked': 0, 
                'not_physik_chunked': 1 
            },
            'big': {
                'physik_chunked': 2, 
                'not_physik_chunked': 3 
            },
            'very_small':{  # for local tests
                'physik_chunked': 4,
                'not_physik_chunked': 5
            }
        }
        return cube_refs[self.cube_type][chunk_status]


    def get_cube_shape(self):
        cube_shapes = {
            "very_small": (400, 400, 400),
            "small": (1400, 1400, 1400), 
            "big": (3500, 3500, 3500),
        }
        return cube_shapes[self.cube_type]