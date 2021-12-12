#!/usr/bin/env bash

# How long is job (in minutes)?
#SBATCH --time=65  #45 #min

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=2 --mem-per-cpu=12000

# Name of jobs?
#SBATCH --job-name=recall_scores

# Number jobs to run in parallel
#SBATCH --array=1 #0-1483 # 1483 nSLs = 1483, nParcels = 181 in HCP-MMP 

# Where to output log files?
#SBATCH -o /jukebox/norman/rmasis/clones/SchemaBigFiles/draft_PAPER/LOGS/slurm_-%j.log

date=$1 
hem=$2 # L or R
roi=$3 # mPFC, SFG, hippo, etc
roi_id=9999
scoretype=roi_recall_score  #roi_recall_score #$4
modality_mode=$4 #within_modality or across_modality

echo "DATE LABEL" $date
echo "HEM" $hem
echo "ROI" $roi
echo "ROI_ID" $roi_id
echo "SCORETYPE" $scoretype

module load pyger/0.8 #/beta 

export HDF5_USE_FILE_LOCKING=FALSE

python 02_RecallStorySchema_step2.py $date $hem $roi $roi_id $scoretype $modality_mode

