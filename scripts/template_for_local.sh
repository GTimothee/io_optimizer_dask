workspace=''
dask_module='dask'
optimizer_module='io_optimizer_dask/'
benchmark_module='io_optimizer_dask/benchmarks/'
dask_utils_perso='dask_utils_perso/'
sources='src/'

export PYTHONPATH="${workspace}${dask_module}:${workspace}${optimizer_module}:${workspace}${dask_utils_perso}:${workspace}${optimizer_module}${sources}:${workspace}${benchmark_module}"
export DATA_PATH=''
export HDD_PATH=''
export OUTPUT_DIR=''
export OUTPUT_BENCHMARK_DIR=''
export BENCHMARK_DIR=""

if [ $1 == "--notebook" ]
then 
    pipenv run jupyter notebook
elif [ $1 == '--test-benchmark' ]
then
    pipenv run pytest benchmarks/tests
elif [ $1 == '--test' ]
then
    pipenv run pytest tests
elif [ $1 == '--speed_tests' ]
then
    pipenv run python benchmarks/benchmarks/speed_tests.py
elif [ $1 == '--exp' ]
then
    pipenv run python benchmarks/benchmarks/experiment_1.py
else
    echo "No arguments. Aborting."
fi