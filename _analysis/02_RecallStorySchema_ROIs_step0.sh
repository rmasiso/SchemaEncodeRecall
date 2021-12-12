#!/bin/bash

############## 
############## STORY AND SCHEMA SCORES at ENCODING for a priori ROIs
############## 

rois=( PMC mPFC PHC Ang SFG )

## uncomment this line when re-running analyses for mPFC clusters
# rois=( mPFC_k2_c0 mPFC_k2_c1 )

hems=( L R )

date=$1 #date identifier

modality_mode=$2 #within/or across modality


############## cortical ROIs

for hem in ${hems[*]}; do
    for roi in ${rois[*]}; do

       sbatch 02_RecallStorySchema_ROIs_step1.sh $date $hem $roi $modality_mode

    done
done

############## HIPPOCAMPAL ROIs

# for roi in ahippo phippo hippo; do

#    sbatch 02_RecallStorySchema_ROIs_step1.sh $date None $roi $modality_mode
   
# done

## don't run hippo ROIs if running mPFC subclusters
if [ ${#rois[@]} -eq 2 ]; then
    echo "Not Running hippocampal ROIs."
else
    echo "Running hippocampal ROIs."
    for roi in ahippo phippo hippo; do
        sbatch 02_RecallStorySchema_ROIs_step1.sh $date None $roi $modality_mode
    done
fi
