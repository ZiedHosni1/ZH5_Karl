#!/bin/bash -l

#$ -l h_rt=24:00:00
#$ -l mem=4G
#$ -l tmpfs=10G
#$ -N tg
#$ -pe smp 4
#$ -wd /home/ucaqkin/Scratch/nsci0017/

EXPERIMENTS_BASE_DIR="${PWD}/tg_jobs" # Assumes -wd is project base, stores experiments subdir here

# Unique identifier for this run (Using Job ID and Date)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# JOB_ID is provided by the SGE scheduler (used by qsub)
RUN_ID="run_${JOB_ID}_${TIMESTAMP}" # Example: run_12345_20250422_173000

OUTPUT_DIR="${EXPERIMENTS_BASE_DIR}/${RUN_ID}"

# Create the output directory safely
mkdir -p "${OUTPUT_DIR}"
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "Error: Could not create output directory ${OUTPUT_DIR}"
    exit 1
fi

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

CODE_DIR="${PWD}code/tg"
cp ${CODE_DIR}/SA_score.pkl.gz "${OUTPUT_DIR}/SA_score.pkl.gz"

# Full training incl. gen and dis pretrain, and followed by adv train
python "${CODE_DIR}/main.py" \
	--output_dir "${OUTPUT_DIR}" \
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
