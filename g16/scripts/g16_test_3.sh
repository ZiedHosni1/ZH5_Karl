#!/bin/bash -l
#$ -N g16_singlejob
#$ -l h_rt=48:00:00
#$ -l mem=8G
#$ -pe smp 4
#$ -l tmpfs=20G

# Submit as `qsub submit_g16_single.sh my_molecule`, make sure not to include .gjf!

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
module load vmd/1.9.3/text-only
source $g16root/g16/bsd/g16.profile # Source Gaussian profile
# module load ghostscript/9.19


# --- Job Execution ---

# Navigate to the temporary directory for execution
cd "$TMPDIR"
echo "Changed working directory to $PWD (TMPDIR)"

# Copy the input file from the submission directory to TMPDIR
# Assumes you run qsub from the directory containing your .gjf file
echo "Copying input file ${INPUT_NAME}.gjf from $SGE_O_WORKDIR to $TMPDIR"
cp "$SGE_O_WORKDIR/${INPUT_NAME}.gjf" .

# Run Gaussian 16
echo "Starting Gaussian 16 calculation..."
# g16 "${INPUT_NAME}.gjf"
g16 "${INPUT_NAME}.gjf"

# Check if Gaussian finished successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Gaussian (g16) failed. Check log file if possible, or SGE error file."
    # Attempt to copy whatever might exist for debugging
    echo "Copying available files from $TMPDIR to $JOB_RESULT_DIR"
    rsync -av --ignore-missing-args ./*.log ./*.chk ./*.fchk ./*.gjf "$JOB_RESULT_DIR/" || echo "rsync failed to copy logs/chk."
    # Copy SGE files if possible (paths might not be valid if job failed very early)
    if [ -n "$SGE_STDOUT_PATH" ] && [ -f "$SGE_STDOUT_PATH" ]; then cp "$SGE_STDOUT_PATH" "$JOB_RESULT_DIR/"; fi
    if [ -n "$SGE_STDERR_PATH" ] && [ -f "$SGE_STDERR_PATH" ]; then cp "$SGE_STDERR_PATH" "$JOB_RESULT_DIR/"; fi
    exit 1 # Exit the script because g16 failed
fi

echo "Gaussian 16 calculation finished."

# --- Post-Processing (formchk, cubegen, VMD plots) ---

echo "Starting post-processing: formchk, cubegen, VMD plots..."

# --- Use formchk to generate .fchk ---
# Check if binary checkpoint file exists and is non-empty
if [ ! -s "${INPUT_NAME}.chk" ]; then
    echo "ERROR: Binary checkpoint file (${INPUT_NAME}.chk) not found or empty. Cannot run formchk."
    # Attempt to copy whatever might exist for debugging
    echo "Copying available files from $TMPDIR to $JOB_RESULT_DIR"
    rsync -av --ignore-missing-args ./*.log ./*.chk ./*.gjf energies.txt "$JOB_RESULT_DIR/"
    if [ -n "$SGE_STDOUT_PATH" ] && [ -f "$SGE_STDOUT_PATH" ]; then cp "$SGE_STDOUT_PATH" "$JOB_RESULT_DIR/"; fi
    if [ -n "$SGE_STDERR_PATH" ] && [ -f "$SGE_STDERR_PATH" ]; then cp "$SGE_STDERR_PATH" "$JOB_RESULT_DIR/"; fi
    exit 1
fi
echo "Binary checkpoint file found. Running formchk..."
formchk -3 "${INPUT_NAME}.chk" "${INPUT_NAME}.fchk"
if [ $? -ne 0 ]; then
    echo "ERROR: formchk failed. Cannot proceed."
    # Attempt to copy whatever might exist for debugging
    echo "Copying available files from $TMPDIR to $JOB_RESULT_DIR"
    rsync -av --ignore-missing-args ./*.log ./*.chk ./*.fchk ./*.gjf energies.txt "$JOB_RESULT_DIR/"
    if [ -n "$SGE_STDOUT_PATH" ] && [ -f "$SGE_STDOUT_PATH" ]; then cp "$SGE_STDOUT_PATH" "$JOB_RESULT_DIR/"; fi
    if [ -n "$SGE_STDERR_PATH" ] && [ -f "$SGE_STDERR_PATH" ]; then cp "$SGE_STDERR_PATH" "$JOB_RESULT_DIR/"; fi
    exit 1
fi
echo "formchk successful."
# --- End of formchk section ---


# Check if fchk exists and is non-empty before proceeding
if [ -s "${INPUT_NAME}.fchk" ]; then # Check for non-zero size
    echo "Formatted checkpoint file found. Running cubegen..."
    # Use NSLOTS for the number of processors in cubegen
    cubegen $NSLOTS MO=HOMO "${INPUT_NAME}.fchk" HOMO.cube 0 h
    if [ $? -ne 0 ]; then echo "Warning: cubegen failed for HOMO"; fi
    cubegen $NSLOTS MO=LUMO "${INPUT_NAME}.fchk" LUMO.cube 0 h
    if [ $? -ne 0 ]; then echo "Warning: cubegen failed for LUMO"; fi
    cubegen $NSLOTS potential=SCF "${INPUT_NAME}.fchk" ESP.cube 0 h
    if [ $? -ne 0 ]; then echo "Warning: cubegen failed for ESP"; fi
    cubegen $NSLOTS density=ELF "${INPUT_NAME}.fchk" ELF.cube 0 h
    if [ $? -ne 0 ]; then echo "Warning: cubegen failed for ELF (Ensure Density=Current or similar was used in G16)"; fi

    # --- Load default ghostscript AFTER gaussian/vmd ---
    # This is loaded here to minimize conflicts earlier
    echo "Loading default ghostscript module..."
    module load ghostscript/9.19
    # -------------------------------------------------

    echo "Generating plots using VMD..."
    # Loop through properties for combined molecule + orbital plots
    for prop in HOMO LUMO ESP ELF; do
        if [ -f "${prop}.cube" ]; then
            echo "  Generating plot for $prop..."
            # Create VMD Tcl script for the combined plot
            cat >vmd_${prop}.tcl <<EOF
# Load the cube file (volumetric data)
mol new ${prop}.cube type cube waitfor all
# Add the geometry from the fchk file to the same molecule ID
mol addfile ${INPUT_NAME}.fchk type fchk waitfor all

# Set background color to white
color Display Background white

# Set representation for the orbital/property (Isosurface)
mol delrep 0 top
mol representation Isosurface 0.03 0 0 1 1 1
mol color Volume
mol addrep top

# Add representation for the molecule itself (e.g., Licorice)
mol representation Licorice 0.100000 12.000000 12.000000
mol color Element
mol addrep top

# Set view and render
display projection Orthographic
render TachyonInternal ${prop}.eps
quit
EOF
            # Run VMD in text mode for the orbital plot
            vmd -dispdev text -e vmd_${prop}.tcl
            if [ $? -ne 0 ]; then
                echo "Warning: VMD failed for $prop plot generation."
            fi

            # Convert EPS to PDF and SVG using Ghostscript (if EPS exists)
            # This now uses the loaded default ghostscript module
            if [ -f "${prop}.eps" ]; then
                echo "  Converting $prop plot to PDF and SVG..."
                gs -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sOutputFile=${prop}.pdf ${prop}.eps # Revert to gs
                if [ $? -ne 0 ]; then echo "Warning: Ghostscript failed to create ${prop}.pdf"; fi
                # Check if default gs supports SVG, otherwise remove this line
                gs -dBATCH -dNOPAUSE -sDEVICE=svg -sOutputFile=${prop}.svg ${prop}.eps # Revert to gs
                if [ $? -ne 0 ]; then echo "Warning: Ghostscript failed to create ${prop}.svg (SVG device might not be supported by default gs)"; fi
            else
                 echo "Warning: ${prop}.eps was not created by VMD. Skipping PDF/SVG conversion for $prop."
            fi
        else
            echo "  Skipping plot for $prop: ${prop}.cube not found (cubegen likely failed)."
        fi
    done # End of property loop
    echo "Orbital/Property VMD plotting finished."

    # --- Block for molecule-only plot (AFTER the loop) ---
    echo "Generating plot for molecule structure..."
    # Create VMD Tcl script for the molecule
    cat >vmd_molecule.tcl <<EOF
# Load the geometry from the fchk file
mol new ${INPUT_NAME}.fchk type fchk waitfor all

# Set background color to white
color Display Background white

# Set representation for the molecule (e.g., Licorice)
mol delrep 0 top
mol representation Licorice 0.100000 12.000000 12.000000
mol color Element
mol addrep top

# Set view and render
display projection Orthographic
render TachyonInternal molecule.eps
quit
EOF
    # Run VMD in text mode
    echo "  Running VMD for molecule structure..."
    vmd -dispdev text -e vmd_molecule.tcl
    # Check VMD exit status
    if [ $? -ne 0 ]; then
        echo "Warning: VMD failed for molecule plot generation."
    fi

    # Convert EPS to PDF and SVG using Ghostscript (if EPS exists)
    # This now uses the loaded default ghostscript module
    if [ -f "molecule.eps" ]; then
        echo "  Converting molecule plot to PDF and SVG..."
        gs -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sOutputFile=molecule.pdf molecule.eps # Revert to gs
        if [ $? -ne 0 ]; then echo "Warning: Ghostscript failed to create molecule.pdf"; fi
         # Check if default gs supports SVG, otherwise remove this line
        gs -dBATCH -dNOPAUSE -sDEVICE=svg -sOutputFile=molecule.svg molecule.eps # Revert to gs
        if [ $? -ne 0 ]; then echo "Warning: Ghostscript failed to create molecule.svg (SVG device might not be supported by default gs)"; fi
    else
        echo "Warning: molecule.eps was not created by VMD. Skipping PDF/SVG conversion."
    fi
    # --- End of added block for molecule-only plot ---

# This 'else' matches the main 'if [ -s "${INPUT_NAME}.fchk" ]' check
else
    echo "Skipping post-processing (cubegen, VMD) because ${INPUT_NAME}.fchk was not found or is empty (or formchk failed)."
fi # This 'fi' closes the main post-processing block check

# --- End of Post-Processing Section ---


# --- Copy Results ---

echo "Copying results from $TMPDIR to $JOB_RESULT_DIR"
# Copy input, log, fchk, cube files, plots, and SGE output/error files
# Using rsync -av ensures verbose output and archive mode (preserves permissions etc.)
rsync -av --ignore-missing-args \
          "${INPUT_NAME}.gjf" \
          "${INPUT_NAME}.log" \
          "${INPUT_NAME}.fchk" \
          energies.txt \
          *.cube \
          *.eps \
          *.pdf \
          *.svg \
          molecule.* \
          "$JOB_RESULT_DIR/"


# Also copy the SGE output/error files for reference
cp "$SGE_STDOUT_PATH" "$JOB_RESULT_DIR/" 2>/dev/null || echo "Warning: Failed to copy SGE stdout."
cp "$SGE_STDERR_PATH" "$JOB_RESULT_DIR/" 2>/dev/null || echo "Warning: Failed to copy SGE stderr."


echo "Job finished at $(date)"

exit 0
