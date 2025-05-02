#!/bin/bash -l

#$ -l h_rt=48:00:00
#$ -l mem=6G
#$ -l tmpfs=10G
#$ -pe smp 4
#$ -t 1-5                       # Set up a job array with tasks numbered 1 to 15
#$ -N tg_hpo_array
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
TASK_RESULT_DIR="${RESULTS_BASE_DIR}/job_${JOB_ID}_task_${SGE_TASK_ID}"
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
# module load python/3.7.0
module load python/miniconda3/24.3.0-0

# Initialise conda.
source $UCL_CONDA_PATH/etc/profile.d/conda.sh

# Activate the conda environment.
conda activate py36tf


# --- Parameter Extraction ---

# Read the parameters for the current task ID from the CSV file
# Use SGE_TASK_ID + 1 to skip the header row
LINE_NUM=$(($SGE_TASK_ID + 1))
echo "Reading parameters from line $LINE_NUM of $PARAM_FILE"

# Get header line
HEADER=$(head -n 1 "$PARAM_FILE")
# Get data line for this task
PARAM_LINE=$(sed -n "${LINE_NUM}p" "$PARAM_FILE")

# Function to get value by header name
get_param() {
  local header_name=$1
    echo "$PARAM_LINE" | awk -F',' -v header="$HEADER" -v field_name="$header_name" '
    BEGIN {
        split(header, headers, ",");
        for (i=1; i<=length(headers); i++) {
            if (headers[i] == field_name) {
                col_num=i;
                break;
            }
        }
    }
    { print $col_num }'
}

# Extract required parameters using the function
ADV_LR=$(get_param "adv_lr")
DIS_LAMBDA=$(get_param "dis_lambda")
GEN_D_MODEL=$(get_param "gen_d_model")
DIS_D_MODEL=$(get_param "dis_d_model")
GEN_NUM_ENCODER_LAYERS=$(get_param "gen_num_encoder_layers")
ROLL_NUM=$(get_param "roll_num")
GEN_DROPOUT=$(get_param "gen_dropout")
BATCH_SIZE=$(get_param "batch_size")

# Check if parameters were extracted successfully
if [ -z "$ADV_LR" ] || [ -z "$DIS_LAMBDA" ] || [ -z "$BATCH_SIZE" ]; then
    echo "ERROR: Failed to extract one or more parameters from $PARAM_FILE for task $SGE_TASK_ID."
    exit 1
fi

echo "Parameters for Task $SGE_TASK_ID:"
echo "  adv_lr: $ADV_LR"
echo "  dis_lambda: $DIS_LAMBDA"
echo "  batch_size: $BATCH_SIZE"


# --- Run TenGAN ---

echo "Starting TenGAN training..."
# Construct the python command with extracted parameters
# Add all necessary flags based on HPO_init_randomSearch.csv and main.py requirements
python main.py \
    --dataset_name comb_1 \
    --properties synthesizability \
    --max_len 110 \
    --batch_size "$BATCH_SIZE" \
    --gen_pretrain \
    --dis_pretrain \
    --adversarial_train \
    --dis_lambda "$DIS_LAMBDA" \
    --adv_lr "$ADV_LR" \
    --gen_d_model "$GEN_D_MODEL" \
    --dis_d_model "$DIS_D_MODEL" \
    --gen_num_encoder_layers "$GEN_NUM_ENCODER_LAYERS" \
    --roll_num "$ROLL_NUM" \
    --gen_dropout "$GEN_DROPOUT" \
    --adv_epochs 150 \
    --gen_train_size 3236 \
    --generated_num 2000

# Check if python script executed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Python script failed for task $SGE_TASK_ID."
    # Optionally copy partial results or logs before exiting
    echo "Copying logs and any partial results from $TMPDIR to $TASK_RESULT_DIR"
    # Use rsync for potentially large/many files, handle potential errors
    rsync -av --ignore-errors "$TMPDIR/" "$TASK_RESULT_DIR/"
    exit 1
fi

echo "TenGAN training finished successfully."


# --- Copy Results ---

echo "Copying results from $TMPDIR to $TASK_RESULT_DIR"
# Modify this if TenGAN saves output elsewhere or you want to copy specific files
# Using rsync is generally robust
rsync -av "$TMPDIR/tg/" "$TASK_RESULT_DIR/"
# Also copy the SGE output/error files for reference
cp "$SGE_STDOUT_PATH" "$TASK_RESULT_DIR/"
cp "$SGE_STDERR_PATH" "$TASK_RESULT_DIR/"

echo "Job finished at $(date)"

exit 0
