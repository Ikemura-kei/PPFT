#! /bin/bash

#SBATCH --gpus 2
#SBATCH -t 00-10:00:00

source ~/.bashrc
conda activate ppft

PROJ_DIR=/home/x_keiik/154_ws/PDNE

cd ${PROJ_DIR} && bash ./scripts/runs/train.sh PPFTFreeze