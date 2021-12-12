#!/usr/bin/env bash

# How long is job (in minutes)?
#SBATCH --time=120 #min

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=1 --mem-per-cpu=12000

# Name of jobs?
#SBATCH --job-name=seedtosl

# Number jobs to run in parallel
#SBATCH --array=1 #0-1483 # 1483 nSLs = 1483, nParcels = 181 in HCP-MMP 

# Where to output log files?
#SBATCH -o /jukebox/norman/rmasis/clones/SchemaBigFiles/draft_PAPER/LOGS/slurm_-%j.log

date=$1 #20200626
seed_roi=$2 #'mPFC'
seed_scoretype=$3 #'roi_recall_score' #the seed
seed_eff=$4 #'story_effect'
post_roi=$5 #'SL' # could also be atlas, 
post_scoretype=$6  #'sl_recall_score'
post_eff=$7 #'schema_effect'

echo "DATE LABEL" $date
echo "seed" $seed_roi
echo "seed scoretype" $seed_scoretype
echo "seed eff" $seed_eff
echo "post scoretype" $post_scoretype
echo "post eff" $post_eff

module load pyger/0.8 #/beta 

export HDF5_USE_FILE_LOCKING=FALSE

python 03S1_SeedtoSL_step2.py $date $seed_roi $seed_scoretype $seed_eff $post_roi $post_scoretype $post_eff

