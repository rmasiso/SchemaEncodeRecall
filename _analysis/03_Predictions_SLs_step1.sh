#!/usr/bin/env bash

# How long is job (in minutes)?
#SBATCH --time=15 #min

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=1 --mem-per-cpu=12000

# Name of jobs?
#SBATCH --job-name=predict

# Number jobs to run in parallel
#SBATCH --array=0-1483 #nSLs = 1483, nParcels = 181 in HCP-MMP 

# Where to output log files?
#SBATCH -o /jukebox/norman/rmasis/clones/SchemaBigFiles/draft_PAPER/LOGS/slurm_-%j.log

### SLs
date=$1
hem=$2 #L #$1 #R #$2
roi=SL #SL #$2 #SL #$3
roi_id=$SLURM_ARRAY_TASK_ID #9999 #$SLURM_ARRAY_TASK_ID #9999 
scoretype=$3 #sl_percept_score #sl_recall_score #$4
extra=$4

# ### ROIS 
# date=20210930 
# hem=$1 #L #$1 #R #$2
# roi=$2 #SL #$2 #SL #$3
# roi_id=9999 #$SLURM_ARRAY_TASK_ID #9999 #$SLURM_ARRAY_TASK_ID #9999 
# scoretype=$3 #sl_percept_score #sl_recall_score #$4
# extra=_WithinSubj



echo "DATE LABEL" $date
echo "HEM" $hem
echo "ROI" $roi
echo "ROI_ID" $roi_id
echo "SCORETYPE" $scoretype
echo "EXTRA" $extra

module load pyger/0.8 #/beta 

export HDF5_USE_FILE_LOCKING=FALSE

python 03_Predictions_step2.py $date $hem $roi $roi_id $scoretype $extra


