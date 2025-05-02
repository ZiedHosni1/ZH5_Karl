#!/bin/bash -l

#$ -l h_rt=48:00:00
#$ -l mem=6G
#$ -l tmpfs=10G
#$ -pe smp 4
#$ -N tg_manual
#$ -wd /home/ucaqkin/Scratch/nsci0017/tg_jobs


# --- Setup ---

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $JOB_ID, Task ID: $SGE_TASK_ID"

# Define paths
PARAM_FILE="/home/ucaqkin/Scratch/nsci0017/code/myriad/HPO_init_randomSearch.csv"
SOURCE_CODE_DIR="/home/ucaqkin/Scratch/nsci0017/code/tg"
# Base directory for storing results in Scratch
RESULTS_BASE_DIR="/home/ucaqkin/Scratch/nsci0017/tg_jobs"

# Create a unique directory for this specific job task's results in Scratch
# This directory will persist after the job finishes
TASK_RESULT_DIR="${RESULTS_BASE_DIR}/job_${JOB_ID}"
mkdir -p "$TASK_RESULT_DIR"
echo "Results will be saved to: $TASK_RESULT_DIR"

# Copy source code to the temporary directory
echo "Copying source code from $SOURCE_CODE_DIR to $TMPDIR"
cp -r "$SOURCE_CODE_DIR" "$TMPDIR"
cd "$TMPDIR/tg"
echo "Changed working directory to $PWD"


# Load required modules.
echo "Loading modules..."
module purge
module load default-modules
module remove compilers mpi
module load python/miniconda3/24.3.0-0

# Initialise conda.
source $UCL_CONDA_PATH/etc/profile.d/conda.sh

# Activate the conda environment.
conda activate py36tf


echo "Starting TenGAN training..."
# Construct the python command with manually set parameters
python main.py \
    --dataset_name comb_1 \
    --properties synthesizability \
    --max_len 100 \
    --batch_size 64 \
    --gen_pretrain \
    --dis_pretrain \
    --adversarial_train \
    --dis_lambda 0.5 \
    --roll_num 8 \
    --adv_epochs 100 \
    --gen_train_size 3236 \
    --generated_num 3500

# Check if python script executed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Python script failed for job $JOB_ID."
    # Optionally copy partial results or logs before exiting
    echo "Copying logs and any partial results from $TMPDIR to $TASK_RESULT_DIR"
    # Use rsync for potentially large/many files, handle potential errors
    rsync -av --ignore-errors "$TMPDIR/tg/" "$TASK_RESULT_DIR/" # Copy from the tg subdirectory in TMPDIR
    # Copy SGE output/error files as well for debugging
    cp "$SGE_STDOUT_PATH" "$TASK_RESULT_DIR/"
    cp "$SGE_STDERR_PATH" "$TASK_RESULT_DIR/"
    exit 1
fi

echo "TenGAN training finished successfully."


# --- Copy Results ---

echo "Copying results from $TMPDIR/tg to $TASK_RESULT_DIR"
# Using rsync is generally robust
rsync -av "$TMPDIR/tg/" "$TASK_RESULT_DIR/"
# Also copy the SGE output/error files for reference
cp "$SGE_STDOUT_PATH" "$TASK_RESULT_DIR/"
cp "$SGE_STDERR_PATH" "$TASK_RESULT_DIR/"

echo "Job finished at $(date)"

exit 0
