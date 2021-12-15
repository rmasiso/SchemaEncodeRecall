# Schema representations in distinct brain networks support narrative memory during encoding and retrieval
Code for the paper *Masís-Obando, Norman, & Baldassano. Schema representations in distinct brain networks support narrative memory during encoding and retrieval*

## Project Organization
--- insert file tree here

## Data

Both encoding and recall neural data in BIDS format is available at [OpenNeuro](https://openneuro.org/datasets/ds001510).

## Preprocessing

After downloading both the encoding and recall data, preprocess and obtain surface data in fsaverage6 resolution. 

### General notes on parallel processing

To run the searchlights and specific ROI analyses, the SLURM scheduler (on servers mantained by the Princeton Neuroscience Institute) was used. Scripts and commands (e.g., “sbatch”) in this repository include the necessary formulations to work on SLURM.

In some of the scripts below, ```module load pyger``` was used to instantiate a virtual environment with python 3.6 installed.

Because the most recent date used to generate the code was 20211014, you will find that the commands in this repository contain this date.

Run code below through instructions chronologically. 


## ======== Main Analyses ========

Instructions for generating the main text results.

## Figure 1 

Memory Scores

1. Run the jupyter notebook ```00_MemoryPerformance_Fig1.ipynb```  to generate the file ```normalizedRubricScores_byStory.pkl``` which contains the quantified memory performance for each story for each subject. 
2. We will need this later for relating the neural scores to behavior. 


## Figure 2 

Story and schema scores for searchlights and specific a priori ROIs

### - GENERATE -

**encoding searchlights**
```
sbatch 02_PerceptionStorySchema_SLs_step1.sh 20211014 L within_modality
sbatch 02_PerceptionStorySchema_SLs_step1.sh 20211014 R within_modality
```

**encoding ROIs**
```
sh 02_PerceptionStorySchema_ROIs_step0.sh 20211014 within_modality
```

**reinstatement searchlights**
```
sbatch 02_RecallStorySchema_SLs_step1.sh 20211014 L within_modality
sbatch 02_RecallStorySchema_SLs_step1.sh 20211014 R within_modality
```
**reinstatement ROIs**
```
sh  02_RecallStorySchema_ROIs_step0.sh 20211014 within_modality
```

### - ANALYZE -

**Searchlight analysis** 
1. Open the jupyter notebook labeled: ```_ImportSearchlightsGenerateMaps.ipynb```
2. Find the cell Labeled **Figure 2**
3. Run to save brainmaps in desired output directory

**ROI analysis** 
1. Open the jupyter notebook labeled: ``` _ImportROIsSeeResults.ipynb```
2. Find the cell labeled **Figure 2**
3. Run to output violin plots 

## Figure 3: 

Predicting behavioral memory for story details with neural measures from encoding and recall.

### - GENERATE -

**encoding searchlights** 
```
sbatch 03_Predictions_SLs_step1.sh 20211014 L sl_percept_score within_modality
sbatch 03_Predictions_SLs_step1.sh 20211014 R sl_percept_score within_modality
```

**reinstatement searchlights**
```
sbatch 03_Predictions_SLs_step1.sh 20211014 L sl_recall_score within_modality
sbatch 03_Predictions_SLs_step1.sh 20211014 R sl_recall_score within_modality
```

**encoding AND reinstatement ROIs:**
1. In ```03_Predictions_ROIs_step0.sh```, make sure that ```rois=( PMC mPFC PHC Ang SFG )```
2. ``sh 03_Predictions_ROIs_step0.sh 20211014 within_modality``

### - ANALYZE -

**Searchlight analysis** 
1. Open the jupyter notebook labeled: ```_ImportSearchlightsGenerateMaps.ipynb```
2. Find the cell Labeled **Figure 3**
3. Run to save brainmaps in desired output directory

**ROI analysis** 
1. Open the jupyter notebook labeled: ``` _ImportROIsSeeResults.ipynb```
2. Find the cell labeled **Figure 3**
3. Run to output violin plots 

## Figure 4 

Regions with significant schema effects and positive associations with subsequent memory.

### - GENERATE -

``sh 03S2_SchemaBenefitModelComparisons_step1.sh 2021014``

### - ANALYZE -


1. In the notebook titled ``ImportSearchlightsGenerateMaps`` find the cell labeled ``Figure 4``
2. Run the cells under it.
3. Files will be saved in the default or (your desired) directory.

## Figure 5

Mediation analysis with mPFC subclusters.

### (1) Generate Clusters

1. First we have to run the silhouette analysis and save out the clusters.
2. Go to the jupyter notebook: ``05_Kmeans.ipynb`` and run all the cells.
3. The clusters will be saved into`` /_data/`` as ``mPFC_c0.h5`` and ``mPFC_c1.h5`` 

### (2) Create Encoding and Reinstatement Story and Schema Scores

Run the encoding and reinstatement story and schema scores:

**ENCODING**

1. Open ``02_PerceptionStorySchema_ROIs_step0.sh``
2. Uncomment the second “rois” line argument that contains ``rois=( mPFC_k2_c0 mPFC_k2_c1 )``
3. Run in terminal with: ``sh 02_PerceptionStorySchema_ROIs_step0.sh 20211014 within_modality``

**REINSTATEMENT**

1. Open ``02_RecallStorySchema_ROIs_step0.sh`` 
2. Uncomment the second “rois” line argument that contains ``rois=( mPFC_k2_c0 mPFC_k2_c1 )``
3. Run in terminal with: ``sh 02_RecallStorySchema_ROIs_step0.sh 20211014 within_modality``

### (3) Use scores in step 2 to predict behavior

1. Open ``03_Predictions_ROIs_step0.sh`` 
2. Uncomment the second “rois” line argument that contains ``rois=( mPFC_k2_c0 mPFC_k2_c1 )``
3. Run in terminal with: ``sh 03_Predictions_ROIs_step0.sh 20211014 within_modality``

### (4) Visualize post-hoc mPFC subclusters results

1. Open ``_ImportROIsSeeResults.ipynb``
2. Find the section titled: ``Figure5; mPFC cluster results``
3. Run the cells to output the bar graphs.

### (5) Run Mediation Analysis

1. Open the jupyter notebook ``05_Mediation.ipynb``
2. Run all the cells




















## ======== Supplementary Analyses ========


Instructions for generating the supplementary results.

## Figure 2 - Supp 1 

Encoding and Reinstatement Story and Schema Within-Subjects

### - GENERATE - 

**encoding searchlights**
```
sbatch 02S1_PerceptionStorySchema_WithinSubj_SLs_step1.sh 20211014 L 
sbatch 02S1_PerceptionStorySchema_WithinSubj_SLs_step1.sh 20211014 R
```

**encoding ROIs**
```
sh 02S1_PerceptionStorySchema_WithinSubj_ROIs_step0.sh 20211014 
```

**reinstatement searchlights**
```
sbatch 02S1_RecallStorySchema_WithinSubj_SLs_step1.sh 20211014 L
sbatch 02S1_RecallStorySchema_WithinSubj_SLs_step1.sh 20211014 R

```
**reinstatement ROIs**
```
sh 02S1_RecallStorySchema_WithinSubj_ROIs_step0.sh 20211014
```

### - ANALYZE - 

**Searchlight Analysis** 
1. Open the jupyter notebook labeled: ``_ ImportSearchlightsGenerateMaps.ipynb``
2. Find the cell Labeled **Figure 2**
3. Uncomment the line ``extra = within_modality_within_subj``
4. Run to save brainmaps in desired output directory

**ROI Analysis** 
1. Open the jupyter notebook labeled:  ``_ImportROIsSeeResults.ipynb``
2. Find the cell labeled **Figure 2**
3. Uncomment the line  ``extra = within_modality_within_subj``
4. Run to output violin plots 

## Figure 2 - Supp 2 

Encoding and Reinstatement Story and Schema Across-Modality

### - GENERATE - 

**encoding searchlights**
```
sbatch 02_PerceptionStorySchema_SLs_step1.sh 20211014 L across_modality
sbatch 02_PerceptionStorySchema_SLs_step1.sh 20211014 R across_modality
```

**reinstatement Searchlights**
```
sbatch 02_RecallStorySchema_SLs_step1.sh 20211014 L across_modality
sbatch 02_RecallStorySchema_SLs_step1.sh 20211014 R across_modality
```

**encoding ROIs**
1. In ``02_PerceptionStorySchema_ROIs_step0.sh``, make sure that ``rois=( PMC mPFC PHC Ang SFG )``
2. ``sh 02_PerceptionStorySchema_ROIs_step0.sh 20211014 across_modality``


**reinstatement ROIs**
1. In ``02_RecallStorySchema_ROIs_step0.sh``, make sure that ``rois=( PMC mPFC PHC Ang SFG )``
2. ``sh  02_RecallStorySchema_ROIs_step0.sh 20211014 across_modality``

### - ANALYZE - 

**Searchlight Analysis** 
1. Open the jupyter notebook labeled: ``_ImportSearchlightsGenerateMaps.ipynb``
2. Find the cell Labeled **Figure 2**
3. Uncomment the line ``extra = ‘across_modality’ ``
4. Run to save brainmaps in desired output directory

**ROI Analysis**
1. Open the jupyter notebook labeled: ``_ImportROIsSeeResults.ipynb``
2. Find the cell labeled **Figure 2**
3. Uncomment the line ``extra = ‘across_modality’``
4. Run to output violin plots 


## Figure 3–Figure Supp 1

Predicting PMC reinstatement story scores with encoding schema scores across cortex.

1. To find associations with PMC’s reinstatement story score and the rest of the brain’s encoding schema score, run the following line:

```
sbatch 03S1_SeedtoSL_step1.sh 20211014 PMC roi_recall_score story_effect SL sl_percept_score schema_effect
```

2. A brainmap will be saved in the default directory ``../../SchemaBigFiles/_PaperOutputData/Brainmaps’ ``

<!-- $date $seed_roi $seed_scoretype $seed_eff $post_roi $post_scoretype $post_eff -->

## Figure 3–Figure Supp 2

Behavioral memory benefit of neural schema scores

### - GENERATE -

```
sh 03S2_SchemaBenefitModelComparisons_step1.sh 
```

Saves searchlight results under a folder called ``sl_percept_score_model_comparisons`` and ``sl_recall_score_model_comparisons`` for the benefit analysis at encoding and retrieval, respectively.

### - ANALYZE -
1. Open the jupyter notebook labeled: ``_ImportSearchlightsGenerateMaps.ipynb``
2. Find the cell labeled: ``Figure 3 - Supplement 2``
3. Run the cell 
4. Brainmap will be saved in ``../../SchemaBigFiles/_PaperOutputData`` under ``model comparisons``

## Figure 3–Figure Supp 3

Predicting behavioral memory for story details with neural measures from encoding and recall within-subjects.

### - GENERATE -

**encoding searchlights**
```
sbatch 03_Predictions_SLs_step1.sh 20211014 L sl_percept_score within_modality_within_subj
sbatch 03_Predictions_SLs_step1.sh 20211014 R sl_percept_score within_modality_within_subj
```

**reinstatement searchlights**
```
sbatch 03_Predictions_SLs_step1.sh 20211014 L sl_recall_score within_modality_within_subj
sbatch 03_Predictions_SLs_step1.sh 20211014 R sl_recall_score within_modality_within_subj
```

**ROIs**

**For both encoding and reinstatement:**
``sh 03_Predictions_ROIs_step0.sh 20211014 within_modality_within_subj``

### - ANALYZE -

**Searchlight ANALYSIS**
1. Open the jupyter notebook labeled: ``_ImportSearchlightsGenerateMaps.ipynb``
2. Find the cell Labeled **Figure 3**
3. Uncomment the line ``extra = 'within_modality_within_subj'``
4. Run to save brainmaps in desired output directory

**ROI ANALYSIS**
1. Open the jupyter notebook labeled:  ``_ImportROIsSeeResults.ipynb``
2. Find the cell labeled **Figure 3**
3. Uncomment the line ``extra = 'within_modality_within_subj'``
4. Run to output violin plots 

### Figure 3–Figure supplement 4

Predicting behavioral memory for story details with neural measures from encoding and recall across-modality.


### - GENERATE -

**encoding searchlights**
```
sbatch 03_Predictions_SLs_step1.sh 20211014 L sl_percept_score across_modality
sbatch 03_Predictions_SLs_step1.sh 20211014 R sl_percept_score across_modality
```

**reinstatement searchlights**
```
sbatch 03_Predictions_SLs_step1.sh 20211014 L sl_recall_score across_modality
sbatch 03_Predictions_SLs_step1.sh 20211014 R sl_recall_score across_modality
```

**ROIs**

**For both encoding and reinstatement:**
1. In ``03_Predictions_ROIs_step0.sh``, make sure that ``rois=( PMC mPFC PHC Ang SFG )``

2. ``sh 03_Predictions_ROIs_step0.sh 20211014 across_modality``

<!--sbatch 03_Predictions_ROIs_step1.sh 20211014 None hippo roi_recall_score across_modality-->

### - ANALYZE -

**Searchlight ANALYSIS**
1. Open the jupyter notebook labeled: ``_ImportSearchlightsGenerateMaps.ipynb``
2. Find the cell Labeled **Figure 3**
3. Uncomment the line ``extra = 'across_modality'``
4. Run to save brainmaps in desired output directory

**ROI ANALYSIS** 
1. Open the jupyter notebook labeled: ``_ImportROIsSeeResults.ipynb``
2. Find the cell labeled **Figure 3**
3. Uncomment the line ``extra = 'across_modality'``
4. Run to output violin plots 

