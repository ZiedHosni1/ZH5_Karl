#!/bin/bash -l

# Batch script to run an OpenMP threaded job under SGE.

module load gaussian/g16-c01/pgi-2018.10
source $g16root/g16/bsd/g16.profile

# Request one hour of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=48:00:00

# Request 1 gigabyte of RAM for each core/thread 
# (must be an integer followed by M, G, or T)
#$ -l mem=4G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=15G

# Set the name of the job.
#$ -N G6

# Request 4 cores.
#$ -pe smp 4

# Set the working directory to somewhere in your scratch space.
#$ -wd /home/ucaqkin/Scratch/Gaussian 

# 8. Run the application.
g16 G6.gjf 
