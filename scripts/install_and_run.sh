mkdir data
mkdir output_imgs
git clone https://github.com/GTimothee/dask_utils_perso.git
git clone https://github.com/GTimothee/dask.git
export PYTHONPATH="$PWD:$PWD/src:$PWD/dask:$PYTHONPATH:$PWD/dask_utils_perso:$PWD/dask_utils_perso/dask_utils_perso"
echo "PYTHONPATH set to $PYTHONPATH"
export DATA_PATH='data'
export OUTPUT_DIR='output_imgs'
pipenv run pytest tests
