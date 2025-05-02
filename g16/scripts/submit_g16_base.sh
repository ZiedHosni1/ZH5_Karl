#!/bin/bash -l
#$ -N g16_base
#$ -l h_rt=48:00:00
#$ -l mem=8G
#$ -pe smp 4
#$ -l tmpfs=10G

# Submit as `qsub submit_g16_base.sh my_molecule`, make sure not to include .gjf!

# Input file name (without .gjf extension) from the first script argument
INPUT_NAME="$1"
echo "Input file base name: $INPUT_NAME"
SCRATCH_OUTPUT_BASE="/home/ucaqkin/Scratch/nsci0017/g16_jobs"

# Create a unique directory for this job's results in Scratch
# This directory will persist after the job finishes
JOB_RESULT_DIR="${SCRATCH_OUTPUT_BASE}/job_${JOB_ID}_${INPUT_NAME}"
mkdir -p "$JOB_RESULT_DIR"
echo "Results will be saved to: $JOB_RESULT_DIR"

# Load necessary modules
echo "Loading modules..."
module purge
module load default-modules
module load gaussian/g16-c01/pgi-2018.10
source $g16root/g16/bsd/g16.profile # Source Gaussian profile


# --- Job Execution ---

# Navigate to the temporary directory for execution
cd "$TMPDIR"
echo "Changed working directory to $PWD (TMPDIR)"

# Copy the input file from the submission directory to TMPDIR
# Assumes you run qsub from the directory containing your .gjf file
echo "Copying input file ${INPUT_NAME}.gjf from $SGE_O_WORKDIR to $TMPDIR"
if [ ! -f "$SGE_O_WORKDIR/${INPUT_NAME}.gjf" ]; then
    echo "Error: Input file not found in submission directory: $SGE_O_WORKDIR/${INPUT_NAME}.gjf"
    # Attempt to copy SGE files if possible
    if [ -n "$JOB_RESULT_DIR" ]; then
        if [ -n "$SGE_STDOUT_PATH" ] && [ -f "$SGE_STDOUT_PATH" ]; then cp "$SGE_STDOUT_PATH" "$JOB_RESULT_DIR/"; fi
        if [ -n "$SGE_STDERR_PATH" ] && [ -f "$SGE_STDERR_PATH" ]; then cp "$SGE_STDERR_PATH" "$JOB_RESULT_DIR/"; fi
    fi
    exit 1
fi
cp "$SGE_O_WORKDIR/${INPUT_NAME}.gjf" .

# Run Gaussian 16, explicitly asking for the formatted checkpoint file
echo "Starting Gaussian 16 calculation..."
g16 -fchk="${INPUT_NAME}.fchk" "${INPUT_NAME}.gjf"

# Check if Gaussian finished successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Gaussian (g16) failed. Check log file if possible, or SGE error file."
    # Attempt to copy whatever might exist for debugging
    echo "Copying available files from $TMPDIR to $JOB_RESULT_DIR"
    # Use rsync to copy, ignore errors if files don't exist yet
    rsync -av --ignore-missing-args \
          "${INPUT_NAME}.gjf" \
          "${INPUT_NAME}.log" \
          "${INPUT_NAME}.chk" \
          "${INPUT_NAME}.fchk" \
          "$JOB_RESULT_DIR/" > /dev/null 2>&1

    # Copy SGE files
    if [ -n "$SGE_STDOUT_PATH" ] && [ -f "$SGE_STDOUT_PATH" ]; then cp "$SGE_STDOUT_PATH" "$JOB_RESULT_DIR/"; fi
    if [ -n "$SGE_STDERR_PATH" ] && [ -f "$SGE_STDERR_PATH" ]; then cp "$SGE_STDERR_PATH" "$JOB_RESULT_DIR/"; fi
    exit 1 # Exit the script because g16 failed
fi

echo "Gaussian 16 calculation finished successfully."


# --- Copy Results ---

echo "Copying results from $TMPDIR to $JOB_RESULT_DIR"
# Copy the essential files: input, log, and the generated fchk
# Checkpoint file (.chk) is usually large and often not needed if fchk is present
rsync -av --ignore-missing-args \
          "${INPUT_NAME}.gjf" \
          "${INPUT_NAME}.log" \
          "${INPUT_NAME}.fchk" \
          "$JOB_RESULT_DIR/"


# Also copy the SGE output/error files for reference
cp "$SGE_STDOUT_PATH" "$JOB_RESULT_DIR/" 2>/dev/null || echo "Warning: Failed to copy SGE stdout."
cp "$SGE_STDERR_PATH" "$JOB_RESULT_DIR/" 2>/dev/null || echo "Warning: Failed to copy SGE stderr."


echo "Job finished at $(date)"

exit 0
