#! /bin/bash

#SBATCH --gpus 1
#SBATCH -t 00-03:50:00

source ~/.bashrc
conda activate ppft

PROJ_DIR=/home/x_keiik/154_ws/PDNE

cd ${PROJ_DIR} && bash ./scripts/runs/debug_test.sh