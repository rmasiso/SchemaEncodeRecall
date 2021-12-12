#!/bin/bash

rois=( MTN MP AT PM PMC mPFC PHC Ang Aud SFG STS mPFC_c0 mPFC_c1 mPFC_c2 PMC_c0 PMC_c1 PMC_c2 )

date=20200626
seed_scoretype=roi_recall_score
seed_eff=story_effect # story_effect or schema_effect
post_roi=SL

for seed_roi in ${rois[*]}; do
    for post_scoretype in sl_percept_score sl_recall_score; do
        for post_eff in story_effect schema_effect; do
           sbatch 20200705_ForPaper_SeedtoSL_step2.sh $date $seed_roi $seed_scoretype $seed_eff $post_roi $post_scoretype $post_eff

        done
    done
done





