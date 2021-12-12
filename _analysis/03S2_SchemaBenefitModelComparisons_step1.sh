#!/bin/bash

# rois=( MTN MP AT PM PMC mPFC PHC Ang Aud SFG STS mPFC_c0 mPFC_c1 mPFC_c2 PMC_c0 PMC_c1 PMC_c2 )

rois=( SL )
hems=( L R )

### FOR SLs (make sure to change array=0-1483
date=$1 #20211014
# hem=$1 #L #$1 #R #$2
# roi=SL #$2 #SL #$2 #SL #$3
# roi_id=$SLURM_ARRAY_TASK_ID #9999 #$SLURM_ARRAY_TASK_ID #9999 
# scoretype=$2 #sl_percept_score #sl_recall_score #$4


for hem in ${hems[*]}; do
    for scoretype in sl_percept_score sl_recall_score; do
        for roi in ${rois[*]}; do
           sbatch 03S2_SchemaBenefitModelComparisons_step2.sh $date $hem $roi $scoretype 

        done
    done
done





