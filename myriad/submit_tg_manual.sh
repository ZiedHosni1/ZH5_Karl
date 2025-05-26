#!/bin/bash -l

#$ -l h_rt=48:00:00
#$ -l mem=6G
#$ -l tmpfs=10G
#$ -pe smp 4
#$ -N tg_manualJob
#$ -wd /home/ucaqkin/Scratch/nsci0017/tg_jobs


# --- Function to copy results ---
cleanup_and_copy() {
    echo "--- TRAP: Job ending signal received or script finished ($1). Copying results ---"
    # Use rsync for robustness, copy only the 'res' directory within tg
    # Add --ignore-errors in case some files are problematic during termination
    # Make sure the target directory exists
    mkdir -p "$TASK_RESULT_DIR/res/" # Ensure target subdir exists
    rsync -av --ignore-errors "$TMPDIR/tg/res/" "$TASK_RESULT_DIR/res/"
    # Also copy SGE output/error files for reference if they exist
    if [ -f "$SGE_STDOUT_PATH" ]; then cp "$SGE_STDOUT_PATH" "$TASK_RESULT_DIR/"; fi
    if [ -f "$SGE_STDERR_PATH" ]; then cp "$SGE_STDERR_PATH" "$TASK_RESULT_DIR/"; fi
    echo "--- TRAP: Copy attempt finished ---"
}

# --- Setup ---

# Define paths
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
module purge
module load default-modules
module remove compilers mpi
module load python/miniconda3/24.3.0-0

# Initialise conda.
source $UCL_CONDA_PATH/etc/profile.d/conda.sh

# Activate the conda environment.
conda activate py36tf


# --- Set Trap ---
# This command will run the 'cleanup_and_copy' function when the script exits for any reason (EXIT)
# or receives common termination signals (TERM, INT, HUP, XCPU).
trap 'cleanup_and_copy $?' EXIT TERM INT HUP XCPU


# Construct the python command with manually set parameters
python main.py \
    --dataset_name comb_1 \
    --properties safscore \
    --max_len 80 \
    --batch_size 64 \
    --gen_pretrain \
    --gen_epochs 50 \
    --dis_pretrain \
    --dis_epochs 20 \
    --adversarial_train \
    --adv_epochs 70 \
    --dis_lambda 0.5 \
    --roll_num 8 \
    --gen_train_size 3236 \
    --generated_num 3600

# Store the exit status of the python script
PYTHON_EXIT_STATUS=$?

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


# --- Generate Loss Plots ---
echo "Generating loss plots..."
python "$TMPDIR/tg/plot_losses.py" --log_dir "$TASK_RESULT_DIR"


# --- Unset Trap ---
# Unset the trap explicitly before exiting normally.
trap - EXIT TERM INT HUP XCPU

echo "Job finished at $(date)"

exit 0
