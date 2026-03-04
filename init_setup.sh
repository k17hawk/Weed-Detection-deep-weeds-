#!/bin/bash

echo "[$(date)]: 'START'"
echo "[$(date)]: 'creating environment with python 3.11 version'"
conda create --prefix ./conda python=3.11 -y
echo "[$(date)]: 'activating the environment'"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./conda
echo "[$(date)]: 'installing the developer requirements'"
pip install -r requirements_dev.txt
echo "[$(date)]: 'END'"
echo "Conda environment: $CONDA_DEFAULT_ENV"