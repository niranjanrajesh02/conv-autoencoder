#! /bin/bash
#PBS -N auto_encoder_tnbc
#PBS -o out.log
#PBS -e err.log
#PBS -l ncpus=50
#PBS -q gpu

source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate tnbc-general
python /home/niranjan.rajesh_ug23/TNBC/conv-autoencoder/main.py