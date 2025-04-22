#!/bin/bash -l

#$ -l h_rt=24:00:00
#$ -l mem=4G
#$ -l tmpfs=10G
#$ -N tg
#$ -pe smp 4
#$ -wd /home/ucaqkin/Scratch/nsci0017/code/tg/

# Load required modules.
# module purge
module load default-modules
module remove compilers mpi
# module load python/3.7.0
module load python/miniconda3/24.3.0-0

# Initialise conda.
source $UCL_CONDA_PATH/etc/profile.d/conda.sh

# Activate the conda environment.
conda activate py36tf

# Full training incl. gen and dis pretrain, and followed by adv train
python main.py \
        --dataset_name comb_1 \
        --properties synthesizability \
        --max_len 120 \
        --batch_size 64 \
        --gen_pretrain \
        --dis_pretrain \
        --adversarial_train \
        --dis_lambda 0.5 \
        --adv_epochs 100 \
        --generated_num 5000
