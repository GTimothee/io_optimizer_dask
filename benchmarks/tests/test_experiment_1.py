import benchmarks

from benchmarks.experiment_1 import *


workspace = os.environ.get('BENCHMARK_DIR')
hdd_path = os.environ.get('HDD_PATH')

def test_load_json(capsys):
    shapes_file_path = workspace + 'chunk_shapes.json'
    d = load_json(shapes_file_path)
    with capsys.disabled():
        print("\nSMALL")
        for _type in d['small']:
            for e in d['small'][_type]:
                print(e)
        
        print("\nBIG")
        for _type in d['big']:
            for e in d['big'][_type]:
                print(e)


def test_create_cube():
    exp_id = 'small'
    for chunked, ref in zip((True, False), range(2)):
        create_cube(hdd_path, ref, exp_id, chunked)