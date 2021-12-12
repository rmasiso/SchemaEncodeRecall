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

### DATE 20201014 for hippo and 20200626 for all ROIS and SLs except hippo

# date=20201014 #$1 #20200626 for all ROIs and SLs except Hippo
# hem=$1 #L #$1 #R #$2
# roi=$2 #SL #$2 #SL #$3
# roi_id=9999 #$SLURM_ARRAY_TASK_ID #9999 #$SLURM_ARRAY_TASK_ID #9999 
# scoretype=$3 #sl_recall_score #$4

# ### FOR ROIs
# date=20200626 #$1 #20200626 for all ROIs and SLs except Hippo
# hem=$1 #L #$1 #R #$2
# roi=$2 #SL #$2 #SL #$3
# roi_id=9999 #$SLURM_ARRAY_TASK_ID #9999 #$SLURM_ARRAY_TASK_ID #9999 
# scoretype=$3 #sl_percept_score #sl_recall_score #$4


### FOR SLs (make sure to change array=0-1483
date=$1 #20200626 #$1 #20200626 for all ROIs and SLs except Hippo
hem=$2 #L #$1 #R #$2
roi=$3 #$2 #SL #$2 #SL #$3
roi_id=$SLURM_ARRAY_TASK_ID #9999 #$SLURM_ARRAY_TASK_ID #9999 
scoretype=$4 #sl_percept_score #sl_recall_score #$4

echo "DATE LABEL" $date
echo "HEM" $hem
echo "ROI" $roi
echo "ROI_ID" $roi_id
echo "SCORETYPE" $scoretype

module load pyger/0.8 #/beta 

export HDF5_USE_FILE_LOCKING=FALSE

python 03S2_SchemaBenefitModelComparisons_step3.py $date $hem $roi $roi_id $scoretype

echo "ran python"

#/usr/bin/time python 20210909_SchemaBenefitModelComparisons_step3.py $date $hem $roi $roi_id $scoretype

