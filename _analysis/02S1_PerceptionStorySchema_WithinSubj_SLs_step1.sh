#!/usr/bin/env bash

# How long is job (in minutes)?
#SBATCH --time=35 #min

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=2 --mem-per-cpu=12000

# Name of jobs?
#SBATCH --job-name=percept_scores

# Number jobs to run in parallel
#SBATCH --array=0-1483 # 1483 nSLs = 1483, nParcels = 181 in HCP-MMP 

# Where to output log files?
#SBATCH -o /jukebox/norman/rmasis/clones/SchemaBigFiles/draft_PAPER/LOGS/slurm_-%j.log



### 20210929 redoing the og SLs
date=$1
hem=$2 #R
roi=SL #SL
roi_id=$SLURM_ARRAY_TASK_ID #9999 
scoretype=sl_percept_score #or any directory name that makes sense
extra=within_modality #


# ### FOR SLs
# date=20200626 
# hem=$1 #R
# roi=SL
# roi_id=$SLURM_ARRAY_TASK_ID #9999 
# scoretype=sl_percept_score #roi_percept_score #or any directory name that makes sense

echo "DATE LABEL" $date
echo "HEM" $hem
echo "ROI" $roi
echo "ROI_ID" $roi_id
echo "SCORETYPE" $scoretype
echo "EXTRA" $extra

module load pyger/0.8 #/beta 

export HDF5_USE_FILE_LOCKING=FALSE

python 02S1_PerceptionStorySchema_WithinSubj_step2.py $date $hem $roi $roi_id $scoretype $extra

