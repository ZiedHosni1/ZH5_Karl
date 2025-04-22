#!/bin/bash
#$ -cwd
#$ -N tg_$JOB_ID
#$ -l h_rt=24:00:00
#$ -l mem=8G
#$ -pe smp 4
#$ -o /home/ucaqkin/Scratch/nsci0017/tg_jobs/$JOB_ID/out.txt
#$ -e /home/ucaqkin/Scratch/nsci0017/tg_jobs/$JOB_ID/err.txt

# 0) Load modules & activate conda env
module load default-modules
module load python/miniconda3/24.3.0-0

# initialise conda
source $UCL_CONDA_PATH/etc/profile.d/conda.sh

# activate the central env (created once, see instructions below)
conda activate py36tf

# 1. unique run directory on Scratch
RUN=/home/ucaqkin/Scratch/nsci0017/tg_jobs/$JOB_ID
mkdir -p $RUN
cp -r /home/ucaqkin/Scratch/nsci0017/code/tg $RUN/code
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
    --pos_file Scratch/nsci0017/data/combined_fuel.csv \
    --save_dir $RUN/results \
    --lightning_logs_dir $RUN/results/lightning_logs

# 5. post‑process → losses.csv & plots
python ../../myriad/tb2csv.py --logdir $RUN/results/lightning_logs \
    --outfile $RUN/losses.csv
python ../../myriad/plot_results.py --run_dir $RUN

# 6. back‑up essentials to AFCS
rsync -av --exclude='*.ckpt' --exclude='events.out*' \
    "$RUN" "$HOME/AFCS/nsci0017/backups_tg/"
