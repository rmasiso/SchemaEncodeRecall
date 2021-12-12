#!/bin/bash

############## 
############## Prediction behavior with Story and Schema scores at encoding or recall
############## 

rois=( PMC mPFC PHC Ang SFG )

## uncomment this line when re-running analyses for mPFC clusters
# rois=( mPFC_k2_c0 mPFC_k2_c1 )

hems=( L R )

scoretypes=( roi_percept_score roi_recall_score) #roi_percept_score,roi_recall_score


date=$1 #date identifier

extra=$2 #within/or across modality or within_modality_within_subj


############## cortical ROIs

for hem in ${hems[*]}; do
    for roi in ${rois[*]}; do
        for scoretype in ${scoretypes[*]}; do

           sbatch 03_Predictions_ROIs_step1.sh $date $hem $roi $scoretype $extra
           
        done

    done
done

############## HIPPOCAMPAL ROIs

# for roi in ahippo phippo hippo; do
#     for scoretype in ${scoretypes[*]}; do
#        sbatch 03_Predictions_ROIs_step1.sh $date None $roi $scoretype $modality_mode       
#     done   
# done


## don't run hippo ROIs if running mPFC subclusters
if [ ${#rois[@]} -eq 2 ]; then
    echo "Not Running hippocampal ROIs."
else
    echo "Running hippocampal ROIs."
    for roi in ahippo phippo hippo; do
        for scoretype in ${scoretypes[*]}; do
           sbatch 03_Predictions_ROIs_step1.sh $date None $roi $scoretype $extra
        done   
    done
fi
