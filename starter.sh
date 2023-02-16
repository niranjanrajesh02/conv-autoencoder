#! /bin/bash
#PBS -N tnbc_conv_ae
#PBS -o out.log
#PBS -e err.log
#PBS -l ncpus=90
#PBS -q cpu

rm /home/niranjan.rajesh_ug23/TNBC/conv-autoencoder/out.log
rm /home/niranjan.rajesh_ug23/TNBC/conv-autoencoder/err.log
rm -r /home/niranjan.rajesh_ug23/TNBC/conv-autoencoder/Results
mkdir /home/niranjan.rajesh_ug23/TNBC/conv-autoencoder/Results
source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate tnbc-general
python /home/niranjan.rajesh_ug23/TNBC/conv-autoencoder/main.py
