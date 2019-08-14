mkdir data
git clone https://github.com/GTimothee/dask_utils_perso.git
git clone https://github.com/GTimothee/dask.git
export PYTHONPATH="$PWD:$PWD/src:$PWD/dask:$PYTHONPATH:"
echo "PYTHONPATH set to $PYTHONPATH"
export DATA_PATH='data'
export OUTPUT_DIR='output_imgs'
pipenv run pytest tests/test_main.py
