#!/bin/bash -l
#$ -N 20250505_01
#$ -l h_rt=24:00:00
#$ -l mem=8G
#$ -pe smp 4
#$ -wd /home/ucaqkin/Scratch/nsci0017

module load gaussian/g16-c01/pgi-2018.10
source $g16root/g16/bsd/g16.profile
module load vmd/1.9.3/text-only

RUN_ROOT=/home/ucaqkin/Scratch/nsci0017/g16_jobs
RUN_DIR="${RUN_ROOT}/${JOB_NAME}_${JOB_ID}"
mkdir -p "$RUN_DIR"

cp "$SGE_O_WORKDIR/g16.gjf" "$RUN_DIR/"
cd "$RUN_DIR"

INPUT="20250505_01"
NPROC=4

g16 $INPUT.gjf
formchk -3 -a "$INPUT.chk" "$INPUT.fchk"
cubegen $NPROC MO=HOMO "$INPUT.fchk" HOMO.cube -50 h
cubegen $NPROC MO=LUMO "$INPUT.fchk" LUMO.cube -50 h
cubegen $NPROC Potential=SCF "$INPUT.fchk" ESP.cube -50 h
cubegen $NPROC Density=SCF "$INPUT.fchk" rho.cube -50 h

for prop in HOMO LUMO ESP; do
    cat >vmd_$prop.tcl <<'EOF'
mol new {{PROP}}.cube type cube waitfor all
mol delrep 0 top
mol representation Isosurface 0.03 0 0 1 1  ;# 0.03 isovalue — adjust if needed
mol color Name
mol addrep top
display projection Orthographic
render TachyonInternal {{PROP}}.eps
quit
EOF
    sed -i "s/{{PROP}}/$prop/g" vmd_$prop.tcl
    vmd -dispdev text -e vmd_$prop.tcl >/dev/null # :contentReference[oaicite:5]{index=5}
    # Convert lightweight EPS → PDF & SVG
    gs -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sOutputFile=${prop}.pdf ${prop}.eps >/dev/null
    gs -dBATCH -dNOPAUSE -sDEVICE=svg -sOutputFile=${prop}.svg ${prop}.eps >/dev/null
done

rsync -av --exclude='chk/' $RUN AFCS/nsci0017/backups_g16/
