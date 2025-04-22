#!/bin/bash
#$ -wd /home/ucaqkin/Scratch/nsci0017
#$ -N tg_$JOB_ID
#$ -l h_rt=24:00:00
#$ -l mem=8G
#$ -pe smp 4

# 0) Load modules & activate conda env
module purge
module load default-modules
module remove compilers mpi
module load python/miniconda3/24.3.0-0

# initialise conda
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate py36tf

# 1. unique run directory on Scratch
RUN=/tg_jobs/$JOB_ID
mkdir -p $RUN
exec >"$RUN/out.txt" 2>"$RUN/err.txt"

cp -r /code/tg $RUN/code
cd $RUN/code

# 2. reproducibility metadata
git rev-parse HEAD >"$RUN/git_hash.txt"
conda env export >"$RUN/env.yml"
echo "$0 $@" >"$RUN/cmd.sh"

# 3. hyper‑parameters (positional args)
DIS_LAMBDA=$1
ARG1=$2

# 4. execute TenGAN
python main.py \
    --dataset_name comb_1 \
    --properties synthesizability \
    --gen_pretrain --dis_pretrain --adversarial_train \
    --dis_lambda 0.5 \
    --adv_epochs 120 \
    --batch_size 64 \
    --max_len 120 \
    --pos_file $HOME/Scratch/nsci0017/data/combined_fuel.csv \
    --save_dir $RUN/results \
    --lightning_logs_dir $RUN/results/lightning_logs

# 5. post‑process → losses.csv & plots
python ../../myriad/tb2csv.py --logdir $RUN/results/lightning_logs \
    --outfile $RUN/losses.csv
python ../../myriad/plot_results.py --run_dir $RUN

# 6. back‑up essentials to AFCS
rsync -av --exclude='*.ckpt' --exclude='events.out*' \
    "$RUN" "$HOME/AFCS/nsci0017/backups_tg/"
