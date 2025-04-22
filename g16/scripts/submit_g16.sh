#!/bin/bash
#$ -cwd
#$ -N g16_$JOB_ID
#$ -l h_rt=12:00:00
#$ -l h_vmem=8G
#$ -pe smp 4
#$ -o Scratch/nsci0017/g16_jobs/${JOB_ID}/out.txt
#$ -e Scratch/nsci0017/g16_jobs/${JOB_ID}/err.txt

###############################################################################
# 1) define absolute scratch path
RUN=$HOME/Scratch/nsci0017/tg_jobs/${JOB_ID}
mkdir -p "$RUN"

# 2) copy code & move in
cp -r $HOME/Scratch/nsci0017/code/tg "$RUN/code"
cd "$RUN/code"

cp $1 $RUN/input.gjf # first arg: path/to.gjf
cd $RUN
export GAUSS_SCRDIR=$RUN/chk

g16 <input.gjf >output.log

# crude xyz extraction
awk 'BEGIN{nat=0} /Standard orientation/{getline;getline;getline;getline}
     NF==6 {print $2,$4,$5,$6}' output.log >output.xyz

rsync -av --exclude='chk/' $RUN AFCS/nsci0017/backups_g16/
