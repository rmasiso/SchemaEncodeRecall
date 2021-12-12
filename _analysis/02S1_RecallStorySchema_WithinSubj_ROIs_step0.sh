#!/bin/bash

############## 
############## STORY AND SCHEMA SCORES at ENCODING for a priori ROIs (WITHIN SUBJECTS)
############## 

rois=( PMC mPFC PHC Ang SFG )

hems=( L R )

date=$1 #date identifier

extra=within_modality


############## cortical ROIs

for hem in ${hems[*]}; do
    for roi in ${rois[*]}; do

       sbatch 02S1_RecallStorySchema_WithinSubj_ROIs_step1.sh $date $hem $roi $extra

    done
done

############## HIPPOCAMPAL ROIs

for roi in ahippo phippo hippo; do

   sbatch 02S1_RecallStorySchema_WithinSubj_ROIs_step1.sh $date None $roi $extra
   
done